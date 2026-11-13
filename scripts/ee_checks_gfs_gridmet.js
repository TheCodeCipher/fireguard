// FireGuard Role 3 -- Earth Engine checks that close the UNKNOWNs in docs/gfs_channel_audit.md (section 8).
// Paste into https://code.earthengine.google.com/ , press Run, then copy the Console output (or screenshot it)
// into docs/evidence/ and fill in the "EE check results" table in docs/gfs_channel_audit.md.
// Nothing here downloads data. Date used: 2018-01-06 (the sample day in docs/data_dictionary.md).

var D = '20180106';
var gfs = ee.ImageCollection('NOAA/GFS0P25');

// ---- CHECK 1: what does the creation code's filter actually select? -------------------------------------
// FirePred.py lines 60-63 compare system:index as a STRING between <D>00F01 and <D>00F24.
// Expected if the dataset is as documented: 24 images, forecast_hours 1..24.
// If the index uses three digits (F001), this returns far more images with forecast_hours up to ~239.
var sel = gfs.filter(ee.Filter.gte('system:index', D + '00F01'))
             .filter(ee.Filter.lte('system:index', D + '00F24'));
print('CHECK 1  number of images selected (expect 24):', sel.size());
print('CHECK 1  system:index values:', sel.aggregate_array('system:index'));
print('CHECK 1  forecast_hours values:', sel.aggregate_array('forecast_hours'));

// ---- CHECK 2: run time, valid time, and when the image was created -----------------------------------------
var first = ee.Image(sel.sort('forecast_hours').first());
var last = ee.Image(sel.sort('forecast_hours', false).first());
function describe(label, img) {
  print(label + ' system:index', img.get('system:index'));
  print(label + ' creation_time (model run)', ee.Date(img.get('creation_time')));
  print(label + ' forecast_time (valid time)', ee.Date(img.get('forecast_time')));
  print(label + ' forecast_hours', img.get('forecast_hours'));
  print(label + ' system:time_start', ee.Date(img.get('system:time_start')));
}
describe('CHECK 2  earliest selected:', first);
describe('CHECK 2  latest selected:', last);
// Level A question: is the latest forecast_time at or after <D+1>T00:00Z ?

// ---- CHECK 3: how does ee.Image.atan2 order its operands, and is it full-circle? ---------------------------
// FirePred.py line 74 is  forecast_v_wind.atan2(forecast_u_wind).
var pt = ee.Geometry.Point([-120, 40]);
function val(img) { return img.reduceRegion(ee.Reducer.first(), pt, 1000).values().get(0); }
print('CHECK 3a  Image(0).atan2(Image(1)): 1.571 => left operand is x; 0 => left operand is y',
      val(ee.Image.constant(0).atan2(ee.Image.constant(1))));
print('CHECK 3b  Image(-1).atan2(Image(-1)): -2.356 => full circle; 0.785 => quadrant-limited',
      val(ee.Image.constant(-1).atan2(ee.Image.constant(-1))));

// ---- CHECK 4: precipitation field meaning on both sides of 2019-11-07 --------------------------------------
// Prints the area-mean of total_precipitation_surface for F01..F12 of a 00Z run. Hourly values look unordered;
// values that grow within F01-F06 and restart at F07 are 6-hour-bucket accumulations; values that only grow
// are run-total accumulations.
function precipSeries(day) {
  var c = gfs.filter(ee.Filter.eq('creation_time', ee.Date(day).millis()))
             .filter(ee.Filter.rangeContains('forecast_hours', 1, 12))
             .sort('forecast_hours');
  var region = ee.Geometry.Rectangle([-125, 32, -104, 49]);  // western US
  // map() over an ImageCollection must return an Image, so attach the mean as a property
  return c.map(function (img) {
    var m = img.select('total_precipitation_surface').reduceRegion(ee.Reducer.mean(), region, 27830);
    return img.set('mean_mm', m.get('total_precipitation_surface'));
  }).aggregate_array('mean_mm');
}
print('CHECK 4  mean precip by forecast hour 1..12, run 2019-02-14 00Z (sum-rule period):', precipSeries('2019-02-14'));
print('CHECK 4  mean precip by forecast hour 1..12, run 2020-02-14 00Z (last-rule period):', precipSeries('2020-02-14'));

// ---- CHECK 5: which 24 hours does the GRIDMET image for day D cover? ---------------------------------------
// FirePred.py lines 49-56 use filterDate(<D>T00:00, <D>T23:59).
var gm = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').filterDate('2018-01-06T00:00', '2018-01-06T23:59');
print('CHECK 5  GRIDMET images selected for the day (expect 1):', gm.size());
var g = ee.Image(gm.first());
print('CHECK 5  system:time_start', ee.Date(g.get('system:time_start')));
print('CHECK 5  system:time_end', ee.Date(g.get('system:time_end')));
print('CHECK 5  all properties (look for status / provisional flags):', g.toDictionary());
// Level A question: is system:time_end later than 2018-01-07T00:00Z (the start of the target day)?
