from fireguard.core.config import get_settings
from fireguard.services.firms_service import FIRMSGateway


def get_firms_gateway() -> FIRMSGateway:
    return FIRMSGateway(get_settings())
