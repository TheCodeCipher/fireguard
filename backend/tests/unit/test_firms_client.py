import asyncio
from datetime import date

import httpx

from fireguard.schemas.firms import FIRMSSource
from fireguard.services.firms_client import FIRMSClient, date_chunks, parse_csv_records
from fireguard.services.firms_service import cache_key, normalize_firms_row


def test_area_date_chunks_never_exceed_five_days():
    chunks = list(date_chunks(date(2026, 1, 1), date(2026, 1, 12)))
    assert chunks == [
        (date(2026, 1, 1), date(2026, 1, 5)),
        (date(2026, 1, 6), date(2026, 1, 10)),
        (date(2026, 1, 11), date(2026, 1, 12)),
    ]
    assert all((end - start).days < 5 for start, end in chunks)


def test_area_chunk_rejects_more_than_five_days():
    try:
        list(date_chunks(date(2026, 1, 1), date(2026, 1, 8), max_days=6))
    except ValueError as exc:
        assert "1 to 5" in str(exc)
    else:
        raise AssertionError("Expected a ValueError")


def test_cache_key_separates_source_platform_and_bbox():
    common = {"bbox": "-125,32,-114,42", "start_date": "2026-01-01"}
    first = cache_key({**common, "source": "VIIRS_SNPP_SP"})
    second = cache_key({**common, "source": "VIIRS_NOAA20_SP"})
    third = cache_key({**common, "source": "VIIRS_SNPP_SP", "bbox": "-120,33,-115,40"})
    assert len({first, second, third}) == 3


def test_normalize_firms_row_preserves_distinct_source():
    row = {
        "latitude": "38.1", "longitude": "-120.2", "bright_ti4": "310.0",
        "frp": "8.5", "confidence": "h", "acq_date": "2026-11-10",
        "acq_time": "945", "satellite": "N", "instrument": "VIIRS", "daynight": "D",
    }
    observation = normalize_firms_row(row, FIRMSSource.VIIRS_SNPP_SP)
    assert observation is not None
    assert observation.source == FIRMSSource.VIIRS_SNPP_SP
    assert observation.platform == "Suomi-NPP"
    assert observation.product == "SP"
    assert observation.acquisition_time == "0945"
    assert observation.confidence.value == "high"


def test_firms_csv_bom_is_removed():
    rows = parse_csv_records(
        "\ufeffdata_id,min_date,max_date\nVIIRS_SNPP_SP,2020-01-01,2026-01-01\n"
    )
    assert rows[0]["data_id"] == "VIIRS_SNPP_SP"


def test_firms_area_request_uses_exact_source_and_chunked_dates():
    requests_seen = []

    async def handler(request: httpx.Request):
        requests_seen.append(str(request.url))
        return httpx.Response(200, text="latitude,longitude,acq_date,acq_time,confidence,frp\n")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as session:
            adapter = FIRMSClient(session, "https://firms.test", "secret")
            result = await adapter.area(
                FIRMSSource.VIIRS_NOAA20_NRT,
                "-125,32,-114,42",
                date(2026, 1, 1),
                date(2026, 1, 6),
            )
            assert result == []

    asyncio.run(run())
    assert len(requests_seen) == 2
    assert all("VIIRS_NOAA20_NRT" in item for item in requests_seen)
    assert any("/5/2026-01-01" in item for item in requests_seen)
    assert any("/1/2026-01-06" in item for item in requests_seen)


def test_upstream_timeout_is_classified_without_exposing_url():
    from fireguard.services.firms_client import FIRMSUpstreamError

    async def handler(_request: httpx.Request):
        raise httpx.ReadTimeout("timed out")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as session:
            adapter = FIRMSClient(session, "https://firms.test", "sensitive-key")
            try:
                await adapter.availability(FIRMSSource.VIIRS_SNPP_SP)
            except FIRMSUpstreamError as exc:
                assert exc.kind == "timeout"
                assert "sensitive-key" not in str(exc)
            else:
                raise AssertionError("Expected upstream timeout classification")

    asyncio.run(run())


def test_cache_key_separates_all_data_affecting_filters():
    common = {
        "source": "VIIRS_SNPP_SP",
        "bbox": "-125,32,-114,42",
        "start_date": "2026-01-01",
        "end_date": "2026-01-05",
        "confidence": None,
        "min_frp": None,
    }
    keys = {
        cache_key(common),
        cache_key({**common, "confidence": "high"}),
        cache_key({**common, "min_frp": 5.0}),
        cache_key({**common, "end_date": "2026-01-04"}),
    }
    assert len(keys) == 4


def test_non_finite_frp_and_brightness_are_normalized_to_none():
    row = {
        "latitude": "38.1", "longitude": "-120.2", "bright_ti4": "nan",
        "frp": "inf", "confidence": "h", "acq_date": "2026-11-10",
        "acq_time": "0945", "instrument": "VIIRS", "daynight": "D",
    }
    observation = normalize_firms_row(row, FIRMSSource.VIIRS_SNPP_SP)
    assert observation is not None
    assert observation.brightness is None
    assert observation.frp is None
