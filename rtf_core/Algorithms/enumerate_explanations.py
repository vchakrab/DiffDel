import csv
import time
from collections import deque
#assigned it a temporary weight for now of 1, what I want to do is give each internal edge a weight too right and then an explanation a weight based on its sum
weighted_dc_example = {(('t1.type', '!=', 't2.type'), ('t1.city', '==', 't2.city'), ('t1.state', '==', 't2.state'), ('t1.municipality', '==', 't2.municipality')) : 1,
                       (('t1.type', '!=', 't2.type'), ('t1.city', '==', 't2.city'), ('t1.state', '==', 't2.state'), ('t1.iata_code', '==', 't2.iata_code')): 1}
airport_constraints = [
    [('t1.id', '==', 't2.id')],
    [('t1.ident', '==', 't2.ident')],
    [('t1.name', '==', 't2.name'), ('t1.elevation_ft', '!=', 't2.elevation_ft'), ('t1.continent', '!=', 't2.continent'), ('t1.iso_country', '==', 't2.iso_country')],
    [('t1.name', '==', 't2.name'), ('t1.continent', '!=', 't2.continent'), ('t1.iso_country', '==', 't2.iso_country'), ('t1.local_code', '!=', 't2.local_code')],
    [('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.continent', '!=', 't2.continent'), ('t1.iso_country', '==', 't2.iso_country'), ('t1.local_code', '!=', 't2.local_code')],
    [('t1.name', '==', 't2.name'), ('t1.latitude_deg', '!=', 't2.latitude_deg'), ('t1.longitude_deg', '==', 't2.longitude_deg'), ('t1.iso_region', '!=', 't2.iso_region')],
    [('t1.name', '==', 't2.name'), ('t1.longitude_deg', '==', 't2.longitude_deg'), ('t1.iso_region', '!=', 't2.iso_region'), ('t1.gps_code', '!=', 't2.gps_code')],
    [('t1.name', '==', 't2.name'), ('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.iso_region', '!=', 't2.iso_region'), ('t1.gps_code', '!=', 't2.gps_code')],
    [('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.iso_country', '==', 't2.iso_country'), ('t1.iso_region', '!=', 't2.iso_region'), ('t1.scheduled_service', '!=', 't2.scheduled_service'), ('t1.iata_code', '==', 't2.iata_code')],
    [('t1.name', '==', 't2.name'), ('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.longitude_deg', '!=', 't2.longitude_deg'), ('t1.elevation_ft', '!=', 't2.elevation_ft'), ('t1.iso_region', '!=', 't2.iso_region')],
    [('t1.longitude_deg', '==', 't2.longitude_deg'), ('t1.continent', '!=', 't2.continent'), ('t1.iso_country', '==', 't2.iso_country')],
    [('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.iso_country', '==', 't2.iso_country'), ('t1.iso_region', '!=', 't2.iso_region'), ('t1.scheduled_service', '!=', 't2.scheduled_service'), ('t1.wikipedia_link', '==', 't2.wikipedia_link')],
    [('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.continent', '!=', 't2.continent'), ('t1.iso_country', '==', 't2.iso_country'), ('t1.gps_code', '!=', 't2.gps_code'), ('t1.keywords', '==', 't2.keywords')],
    [('t1.longitude_deg', '==', 't2.longitude_deg'), ('t1.scheduled_service', '!=', 't2.scheduled_service'), ('t1.iata_code', '==', 't2.iata_code'), ('t1.keywords', '==', 't2.keywords')],
    [('t1.longitude_deg', '==', 't2.longitude_deg'), ('t1.scheduled_service', '!=', 't2.scheduled_service'), ('t1.local_code', '!=', 't2.local_code'), ('t1.wikipedia_link', '==', 't2.wikipedia_link'), ('t1.keywords', '==', 't2.keywords')],
    [('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.iso_region', '!=', 't2.iso_region'), ('t1.scheduled_service', '!=', 't2.scheduled_service'), ('t1.iata_code', '==', 't2.iata_code'), ('t1.keywords', '==', 't2.keywords')],
    [('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.continent', '!=', 't2.continent'), ('t1.iso_country', '==', 't2.iso_country'), ('t1.wikipedia_link', '==', 't2.wikipedia_link'), ('t1.keywords', '==', 't2.keywords')],
    [('t1.iso_country', '!=', 't2.iso_country'), ('t1.iso_region', '==', 't2.iso_region')],
    [('t1.type', '!=', 't2.type'), ('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.continent', '!=', 't2.continent'), ('t1.iso_country', '==', 't2.iso_country'), ('t1.iata_code', '==', 't2.iata_code')],
    [('t1.type', '!=', 't2.type'), ('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.continent', '!=', 't2.continent'), ('t1.iso_country', '==', 't2.iso_country'), ('t1.home_link', '==', 't2.home_link')],
    [('t1.type', '!=', 't2.type'), ('t1.longitude_deg', '==', 't2.longitude_deg'), ('t1.iso_country', '==', 't2.iso_country'), ('t1.scheduled_service', '!=', 't2.scheduled_service'), ('t1.wikipedia_link', '==', 't2.wikipedia_link')],
    [('t1.type', '!=', 't2.type'), ('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.longitude_deg', '==', 't2.longitude_deg'), ('t1.elevation_ft', '!=', 't2.elevation_ft'), ('t1.iso_region', '!=', 't2.iso_region')],
    [('t1.type', '!=', 't2.type'), ('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.continent', '!=', 't2.continent'), ('t1.iso_country', '==', 't2.iso_country'), ('t1.wikipedia_link', '==', 't2.wikipedia_link')],
    [('t1.type', '!=', 't2.type'), ('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.longitude_deg', '==', 't2.longitude_deg'), ('t1.iso_region', '!=', 't2.iso_region'), ('t1.local_code', '!=', 't2.local_code')],
    [('t1.type', '!=', 't2.type'), ('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.longitude_deg', '==', 't2.longitude_deg'), ('t1.iso_region', '!=', 't2.iso_region'), ('t1.iata_code', '==', 't2.iata_code')],
    [('t1.type', '!=', 't2.type'), ('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.longitude_deg', '==', 't2.longitude_deg'), ('t1.iso_region', '!=', 't2.iso_region'), ('t1.home_link', '==', 't2.home_link')],
    [('t1.type', '!=', 't2.type'), ('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.longitude_deg', '==', 't2.longitude_deg'), ('t1.iso_region', '!=', 't2.iso_region'), ('t1.scheduled_service', '==', 't2.scheduled_service')],
    [('t1.type', '!=', 't2.type'), ('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.longitude_deg', '==', 't2.longitude_deg'), ('t1.iso_region', '!=', 't2.iso_region'), ('t1.wikipedia_link', '==', 't2.wikipedia_link')],
    [('t1.type', '!=', 't2.type'), ('t1.longitude_deg', '==', 't2.longitude_deg'), ('t1.scheduled_service', '!=', 't2.scheduled_service'), ('t1.iata_code', '==', 't2.iata_code')],
    [('t1.type', '!=', 't2.type'), ('t1.name', '==', 't2.name'), ('t1.longitude_deg', '==', 't2.longitude_deg'), ('t1.municipality', '!=', 't2.municipality')],
    [('t1.type', '!=', 't2.type'), ('t1.name', '==', 't2.name'), ('t1.longitude_deg', '==', 't2.longitude_deg'), ('t1.elevation_ft', '!=', 't2.elevation_ft')],
    [('t1.type', '!=', 't2.type'), ('t1.name', '==', 't2.name'), ('t1.continent', '!=', 't2.continent'), ('t1.iso_country', '==', 't2.iso_country')],
    [('t1.type', '!=', 't2.type'), ('t1.name', '==', 't2.name'), ('t1.longitude_deg', '==', 't2.longitude_deg'), ('t1.iso_region', '!=', 't2.iso_region')],
    [('t1.type', '!=', 't2.type'), ('t1.name', '==', 't2.name'), ('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.iso_region', '!=', 't2.iso_region')],
    [('t1.type', '!=', 't2.type'), ('t1.name', '==', 't2.name'), ('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.wikipedia_link', '==', 't2.wikipedia_link')],
    [('t1.type', '!=', 't2.type'), ('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.continent', '!=', 't2.continent'), ('t1.iso_country', '==', 't2.iso_country'), ('t1.keywords', '==', 't2.keywords')],
    [('t1.type', '!=', 't2.type'), ('t1.longitude_deg', '==', 't2.longitude_deg'), ('t1.scheduled_service', '!=', 't2.scheduled_service'), ('t1.wikipedia_link', '==', 't2.wikipedia_link'), ('t1.keywords', '==', 't2.keywords')],
    [('t1.type', '!=', 't2.type'), ('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.longitude_deg', '==', 't2.longitude_deg'), ('t1.iso_region', '!=', 't2.iso_region'), ('t1.keywords', '==', 't2.keywords')],
    [('t1.name', '==', 't2.name'), ('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.longitude_deg', '!=', 't2.longitude_deg'), ('t1.elevation_ft', '>', 't2.elevation_ft'), ('t1.iso_region', '!=', 't2.iso_region')],
    [('t1.name', '==', 't2.name'), ('t1.latitude_deg', '==', 't2.latitude_deg'), ('t1.longitude_deg', '!=', 't2.longitude_deg'), ('t1.elevation_ft', '<', 't2.elevation_ft'), ('t1.iso_region', '!=', 't2.iso_region')],
]
nc_voter_constraints = [
    [('t1.state', '!=', 't2.state')],
    [('t1.voter_id', '!=', 't2.voter_id'), ('t1.voter_reg_num', '==', 't2.voter_reg_num')],
    [('t1.voter_id', '==', 't2.voter_id'), ('t1.voter_reg_num', '!=', 't2.voter_reg_num')],
    [('t1.voter_id', '==', 't2.voter_id'), ('t1.last_name', '!=', 't2.last_name')],
    [('t1.voter_reg_num', '==', 't2.voter_reg_num'), ('t1.last_name', '!=', 't2.last_name')],
    [('t1.voter_id', '==', 't2.voter_id'), ('t1.first_name', '!=', 't2.first_name')],
    [('t1.voter_reg_num', '==', 't2.voter_reg_num'), ('t1.first_name', '!=', 't2.first_name')],
    [('t1.street_address', '==', 't2.street_address'), ('t1.zip_code', '!=', 't2.zip_code')],
    [('t1.voter_id', '==', 't2.voter_id'), ('t1.zip_code', '!=', 't2.zip_code')],
    [('t1.voter_reg_num', '==', 't2.voter_reg_num'), ('t1.zip_code', '!=', 't2.zip_code')],
    [('t1.street_address', '==', 't2.street_address'), ('t1.city', '!=', 't2.city')],
    [('t1.voter_id', '==', 't2.voter_id'), ('t1.city', '!=', 't2.city')],
    [('t1.voter_reg_num', '==', 't2.voter_reg_num'), ('t1.city', '!=', 't2.city')],
    [('t1.voter_id', '==', 't2.voter_id'), ('t1.register_date', '!=', 't2.register_date')],
    [('t1.voter_reg_num', '==', 't2.voter_reg_num'), ('t1.register_date', '!=', 't2.register_date')],
    [('t1.voter_id', '==', 't2.voter_id'), ('t1.age', '!=', 't2.age')],
    [('t1.voter_reg_num', '==', 't2.voter_reg_num'), ('t1.age', '!=', 't2.age')],
    [('t1.voter_id', '==', 't2.voter_id'), ('t1.download_month', '==', 't2.download_month')],
    [('t1.voter_reg_num', '==', 't2.voter_reg_num'), ('t1.download_month', '==', 't2.download_month')],
    [('t1.voter_reg_num', '==', 't2.voter_reg_num'), ('t1.full_phone_num', '==', 't2.full_phone_num')],
    [('t1.voter_id', '==', 't2.voter_id'), ('t1.full_phone_num', '==', 't2.full_phone_num')],
    [('t1.voter_id', '==', 't2.voter_id'), ('t1.gender', '!=', 't2.gender')],
    [('t1.voter_reg_num', '==', 't2.voter_reg_num'), ('t1.gender', '!=', 't2.gender')],
    [('t1.voter_reg_num', '==', 't2.voter_reg_num'), ('t1.age', '>', 't2.age')],
    [('t1.voter_id', '==', 't2.voter_id'), ('t1.age', '<', 't2.age')],
    [('t1.voter_id', '==', 't2.voter_id'), ('t1.age', '>', 't2.age')],
    [('t1.voter_reg_num', '==', 't2.voter_reg_num'), ('t1.age', '<', 't2.age')],
    [('t1.voter_id', '==', 't2.voter_id'), ('t1.birth_place', '!=', 't2.birth_place')],
    [('t1.voter_reg_num', '==', 't2.voter_reg_num'), ('t1.birth_place', '!=', 't2.birth_place')],
    [('t1.voter_id', '==', 't2.voter_id'), ('t1.middle_name', '==', 't2.middle_name')],
    [('t1.voter_reg_num', '==', 't2.voter_reg_num'), ('t1.middle_name', '==', 't2.middle_name')],
    [('t1.first_name', '==', 't2.first_name'), ('t1.street_address', '==', 't2.street_address')],
    [('t1.voter_id', '==', 't2.voter_id'), ('t1.street_address', '==', 't2.street_address')],
    [('t1.voter_reg_num', '==', 't2.voter_reg_num'), ('t1.street_address', '==', 't2.street_address')],
    [('t1.last_name', '!=', 't2.last_name'), ('t1.name_suffix', '!=', 't2.name_suffix'), ('t1.street_address', '==', 't2.street_address')],
    [('t1.name_prefix', '!=', 't2.name_prefix'), ('t1.last_name', '!=', 't2.last_name'), ('t1.street_address', '==', 't2.street_address')],
    [('t1.name_prefix', '!=', 't2.name_prefix'), ('t1.city', '!=', 't2.city'), ('t1.zip_code', '==', 't2.zip_code')],
    [('t1.name_suffix', '!=', 't2.name_suffix'), ('t1.age', '!=', 't2.age'), ('t1.street_address', '==', 't2.street_address'), ('t1.register_date', '!=', 't2.register_date')],
    [('t1.age', '!=', 't2.age'), ('t1.race', '!=', 't2.race'), ('t1.ethnic', '==', 't2.ethnic'), ('t1.street_address', '==', 't2.street_address')],
    [('t1.middle_name', '==', 't2.middle_name'), ('t1.last_name', '!=', 't2.last_name'), ('t1.street_address', '==', 't2.street_address'), ('t1.register_date', '!=', 't2.register_date')],
]
hospital_constraints = [    [('t1.measurecode', '!=', 't2.measurecode'), ('t1.MeasureName', '==', 't2.MeasureName')],
    [('t1.measurecode', '==', 't2.measurecode'), ('t1.MeasureName', '!=', 't2.MeasureName')],
    [('t1.ProviderNumber', '==', 't2.ProviderNumber'), ('t1.PhoneNumber', '!=', 't2.PhoneNumber')],
    [('t1.ProviderNumber', '==', 't2.ProviderNumber'), ('t1.HospitalName', '!=', 't2.HospitalName')],
    [('t1.ZIPcode', '!=', 't2.ZIPcode'), ('t1.PhoneNumber', '==', 't2.PhoneNumber')],
    [('t1.ProviderNumber', '==', 't2.ProviderNumber'), ('t1.ZIPcode', '!=', 't2.ZIPcode')],
    [('t1.measurecode', '!=', 't2.measurecode'), ('t1.StateAvg', '==', 't2.StateAvg')],
    [('t1.MeasureName', '!=', 't2.MeasureName'), ('t1.StateAvg', '==', 't2.StateAvg')],
    [('t1.City', '!=', 't2.City'), ('t1.ZIPcode', '==', 't2.ZIPcode')],
    [('t1.City', '!=', 't2.City'), ('t1.PhoneNumber', '==', 't2.PhoneNumber')],
    [('t1.ProviderNumber', '==', 't2.ProviderNumber'), ('t1.City', '!=', 't2.City')],
    [('t1.ProviderNumber', '==', 't2.ProviderNumber'), ('t1.CountyName', '!=', 't2.CountyName')],
    [('t1.ProviderNumber', '==', 't2.ProviderNumber'), ('t1.HospitalOwner', '!=', 't2.HospitalOwner')],
    [('t1.State', '==', 't2.State'), ('t1.MeasureName', '==', 't2.MeasureName'), ('t1.StateAvg', '!=', 't2.StateAvg')],
    [('t1.State', '==', 't2.State'), ('t1.measurecode', '==', 't2.measurecode'), ('t1.StateAvg', '!=', 't2.StateAvg')],
    [('t1.Condition', '!=', 't2.Condition'), ('t1.measurecode', '==', 't2.measurecode')],
    [('t1.Condition', '!=', 't2.Condition'), ('t1.MeasureName', '==', 't2.MeasureName')],
    [('t1.Condition', '!=', 't2.Condition'), ('t1.StateAvg', '==', 't2.StateAvg')],
    [('t1.ProviderNumber', '==', 't2.ProviderNumber'), ('t1.measurecode', '==', 't2.measurecode')],
    [('t1.ProviderNumber', '==', 't2.ProviderNumber'), ('t1.MeasureName', '==', 't2.MeasureName')],
    [('t1.ProviderNumber', '==', 't2.ProviderNumber'), ('t1.StateAvg', '==', 't2.StateAvg')],
    [('t1.ProviderNumber', '==', 't2.ProviderNumber'), ('t1.EmergencyService', '!=', 't2.EmergencyService')],
    [('t1.ZIPcode', '==', 't2.ZIPcode'), ('t1.CountyName', '!=', 't2.CountyName'), ('t1.PhoneNumber', '!=', 't2.PhoneNumber'), ('t1.HospitalOwner', '==', 't2.HospitalOwner')],
    [('t1.HospitalName', '==', 't2.HospitalName'), ('t1.State', '==', 't2.State'), ('t1.HospitalOwner', '!=', 't2.HospitalOwner'), ('t1.EmergencyService', '!=', 't2.EmergencyService')],
    [('t1.ProviderNumber', '!=', 't2.ProviderNumber'), ('t1.HospitalName', '==', 't2.HospitalName'), ('t1.City', '==', 't2.City'), ('t1.State', '==', 't2.State')],
    [('t1.HospitalName', '==', 't2.HospitalName'), ('t1.City', '==', 't2.City'), ('t1.State', '==', 't2.State'), ('t1.PhoneNumber', '!=', 't2.PhoneNumber')],
    [('t1.HospitalName', '==', 't2.HospitalName'), ('t1.City', '==', 't2.City'), ('t1.State', '==', 't2.State'), ('t1.ZIPcode', '!=', 't2.ZIPcode')],
    [('t1.ZIPcode', '==', 't2.ZIPcode'), ('t1.measurecode', '==', 't2.measurecode'), ('t1.StateAvg', '!=', 't2.StateAvg')],
    [('t1.ZIPcode', '==', 't2.ZIPcode'), ('t1.MeasureName', '==', 't2.MeasureName'), ('t1.StateAvg', '!=', 't2.StateAvg')],
    [('t1.PhoneNumber', '==', 't2.PhoneNumber'), ('t1.measurecode', '==', 't2.measurecode'), ('t1.StateAvg', '!=', 't2.StateAvg')],
    [('t1.PhoneNumber', '==', 't2.PhoneNumber'), ('t1.MeasureName', '==', 't2.MeasureName'), ('t1.StateAvg', '!=', 't2.StateAvg')],
    [('t1.ProviderNumber', '!=', 't2.ProviderNumber'), ('t1.HospitalName', '==', 't2.HospitalName'), ('t1.ZIPcode', '==', 't2.ZIPcode')],
    [('t1.HospitalName', '==', 't2.HospitalName'), ('t1.ZIPcode', '==', 't2.ZIPcode'), ('t1.PhoneNumber', '!=', 't2.PhoneNumber')],
    [('t1.ProviderNumber', '!=', 't2.ProviderNumber'), ('t1.HospitalName', '==', 't2.HospitalName'), ('t1.PhoneNumber', '==', 't2.PhoneNumber')],
    [('t1.HospitalName', '==', 't2.HospitalName'), ('t1.City', '==', 't2.City'), ('t1.State', '==', 't2.State'), ('t1.CountyName', '!=', 't2.CountyName')],
    [('t1.HospitalName', '==', 't2.HospitalName'), ('t1.ZIPcode', '==', 't2.ZIPcode'), ('t1.CountyName', '!=', 't2.CountyName')],
    [('t1.HospitalName', '==', 't2.HospitalName'), ('t1.CountyName', '!=', 't2.CountyName'), ('t1.PhoneNumber', '==', 't2.PhoneNumber')],
    [('t1.HospitalName', '==', 't2.HospitalName'), ('t1.State', '==', 't2.State'), ('t1.CountyName', '==', 't2.CountyName'), ('t1.HospitalOwner', '!=', 't2.HospitalOwner')],
    [('t1.HospitalName', '==', 't2.HospitalName'), ('t1.City', '==', 't2.City'), ('t1.State', '==', 't2.State'), ('t1.HospitalOwner', '!=', 't2.HospitalOwner')],
    [('t1.State', '!=', 't2.State'), ('t1.StateAvg', '==', 't2.StateAvg')],
]
tax_constraints = [[('t1.city', '!=', 't2.city'), ('t1.zip', '==', 't2.zip')],
    [('t1.area_code', '==', 't2.area_code'), ('t1.state', '!=', 't2.state')],
    [('t1.state', '!=', 't2.state'), ('t1.zip', '==', 't2.zip')],
    [('t1.single_exemp', '>', 't2.single_exemp'), ('t1.married_exemp', '>', 't2.married_exemp')],
    [('t1.single_exemp', '<', 't2.single_exemp'), ('t1.married_exemp', '<', 't2.married_exemp')],
    [('t1.marital_status', '==', 't2.marital_status'), ('t1.single_exemp', '!=', 't2.single_exemp'), ('t1.married_exemp', '!=', 't2.married_exemp')],
    [('t1.fname', '==', 't2.fname'), ('t1.gender', '!=', 't2.gender')],
    [('t1.area_code', '==', 't2.area_code'), ('t1.phone', '==', 't2.phone')],
    [('t1.phone', '==', 't2.phone'), ('t1.city', '==', 't2.city')],
    [('t1.fname', '==', 't2.fname'), ('t1.phone', '==', 't2.phone')],
    [('t1.lname', '==', 't2.lname'), ('t1.phone', '==', 't2.phone')],
    [('t1.phone', '==', 't2.phone'), ('t1.zip', '==', 't2.zip')],
    [('t1.marital_status', '==', 't2.marital_status'), ('t1.single_exemp', '!=', 't2.single_exemp'), ('t1.married_exemp', '>', 't2.married_exemp')],
    [('t1.marital_status', '==', 't2.marital_status'), ('t1.single_exemp', '!=', 't2.single_exemp'), ('t1.married_exemp', '<', 't2.married_exemp')],
    [('t1.marital_status', '==', 't2.marital_status'), ('t1.single_exemp', '>', 't2.single_exemp'), ('t1.married_exemp', '!=', 't2.married_exemp')],
    [('t1.marital_status', '==', 't2.marital_status'), ('t1.single_exemp', '<', 't2.single_exemp'), ('t1.married_exemp', '!=', 't2.married_exemp')],
    [('t1.state', '==', 't2.state'), ('t1.has_child', '==', 't2.has_child'), ('t1.child_exemp', '!=', 't2.child_exemp')],
    [('t1.area_code', '==', 't2.area_code'), ('t1.has_child', '==', 't2.has_child'), ('t1.child_exemp', '!=', 't2.child_exemp')],
    [('t1.state', '==', 't2.state'), ('t1.marital_status', '==', 't2.marital_status'), ('t1.single_exemp', '!=', 't2.single_exemp')],
    [('t1.state', '==', 't2.state'), ('t1.marital_status', '==', 't2.marital_status'), ('t1.married_exemp', '!=', 't2.married_exemp')],
    [('t1.zip', '==', 't2.zip'), ('t1.has_child', '==', 't2.has_child'), ('t1.child_exemp', '!=', 't2.child_exemp')],
    [('t1.area_code', '==', 't2.area_code'), ('t1.marital_status', '==', 't2.marital_status'), ('t1.single_exemp', '!=', 't2.single_exemp')],
    [('t1.area_code', '==', 't2.area_code'), ('t1.marital_status', '==', 't2.marital_status'), ('t1.married_exemp', '!=', 't2.married_exemp')],
    [('t1.zip', '==', 't2.zip'), ('t1.marital_status', '==', 't2.marital_status'), ('t1.single_exemp', '!=', 't2.single_exemp')],
    [('t1.zip', '==', 't2.zip'), ('t1.marital_status', '==', 't2.marital_status'), ('t1.married_exemp', '!=', 't2.married_exemp')],
    [('t1.state', '==', 't2.state'), ('t1.single_exemp', '!=', 't2.single_exemp'), ('t1.married_exemp', '==', 't2.married_exemp')],
    [('t1.state', '==', 't2.state'), ('t1.single_exemp', '==', 't2.single_exemp'), ('t1.married_exemp', '!=', 't2.married_exemp')],
    [('t1.area_code', '==', 't2.area_code'), ('t1.single_exemp', '!=', 't2.single_exemp'), ('t1.married_exemp', '==', 't2.married_exemp')],
    [('t1.state', '==', 't2.state'), ('t1.salary', '==', 't2.salary'), ('t1.rate', '!=', 't2.rate')],
    [('t1.area_code', '==', 't2.area_code'), ('t1.single_exemp', '==', 't2.single_exemp'), ('t1.married_exemp', '!=', 't2.married_exemp')],
    [('t1.zip', '==', 't2.zip'), ('t1.single_exemp', '!=', 't2.single_exemp'), ('t1.married_exemp', '==', 't2.married_exemp')],
    [('t1.zip', '==', 't2.zip'), ('t1.single_exemp', '==', 't2.single_exemp'), ('t1.married_exemp', '!=', 't2.married_exemp')],
    [('t1.area_code', '==', 't2.area_code'), ('t1.salary', '==', 't2.salary'), ('t1.rate', '!=', 't2.rate')],
    [('t1.zip', '==', 't2.zip'), ('t1.salary', '==', 't2.salary'), ('t1.rate', '!=', 't2.rate')],
    [('t1.marital_status', '==', 't2.marital_status'), ('t1.single_exemp', '<', 't2.single_exemp'), ('t1.married_exemp', '>', 't2.married_exemp')],
    [('t1.marital_status', '==', 't2.marital_status'), ('t1.single_exemp', '>', 't2.single_exemp'), ('t1.married_exemp', '<', 't2.married_exemp')],
    [('t1.state', '==', 't2.state'), ('t1.has_child', '==', 't2.has_child'), ('t1.child_exemp', '>', 't2.child_exemp')],
    [('t1.state', '==', 't2.state'), ('t1.has_child', '==', 't2.has_child'), ('t1.child_exemp', '<', 't2.child_exemp')],
    [('t1.state', '==', 't2.state'), ('t1.marital_status', '==', 't2.marital_status'), ('t1.single_exemp', '>', 't2.single_exemp')],
    [('t1.state', '==', 't2.state'), ('t1.marital_status', '==', 't2.marital_status'), ('t1.single_exemp', '<', 't2.single_exemp')],
]
hospital_attributes = ["ProviderNumber", "HospitalName", "City", "State", "ZIPCode",
                           "CountyName", "PhoneNumber", "HospitalType", "HospitalOwner",
                           "EmergencyService", "Condition", "MeasureCode", "MeasureName", "Sample",
                           "StateAvg"]
tax_attributes = ["fname",
                  "lname", "gender", "area_code", "phone", "city", "state", "zip", "marital_status", "has_child", "salary", "rate", "single_exemp", "married_exemp", "child_exemp"]
nc_voterattributes = [
    "voter_id",
    "voter_reg_num",
    "name_prefix",
    "first_name",
    "middle_name",
    "last_name",
    "name_suffix",
    "age",
    "gender",
    "race",
    "ethnic",
    "street_address",
    "city",
    "state",
    "zip_code",
    "full_phone_num",
    "birth_place",
    "register_date",
    "download_month"
]
airport_attributes = ['ident', 'type', 'name', 'latitude_deg', 'longitude_deg', 'elevation_ft',
                    'iso_country', 'iso_region', 'municipality', 'scheduled_service']


dc = [
    [('t1.type', '!=', 't2.type'), ('t1.city', '==', 't2.city'), ('t1.state', '==', 't2.state'), ('t1.municipality', '==', 't2.municipality')],
    [('t1.type', '!=', 't2.type'), ('t1.city', '==', 't2.city'), ('t1.state', '==', 't2.state'), ('t1.iata_code', '==', 't2.iata_code')]
]
dc_internal_test = [
    # E1: Contains Target ('t1.type') and Internal Cell ('t1.intermediate')
    [('t1.type', '!=', 't2.type'), ('t1.intermediate', '==', 't2.intermediate')],
    # E2: Connects to E1 via 't1.intermediate' and hits Boundary ('t1.city')
    [('t1.intermediate', '!=', 't2.intermediate'), ('t1.city', '==', 't2.city')]
]


def set_edge_weight(hypergraph_list, weights = None):
    weighted_hypergraph_dict = {}

    if weights is None:
        for edge_set in hypergraph_list:
            weighted_hypergraph_dict[frozenset(edge_set)] = 1.0
    else:
        for edge_set, weight in zip(hypergraph_list, weights):
            weighted_hypergraph_dict[frozenset(edge_set)] = float(weight)

    return weighted_hypergraph_dict

def build_graph_data(denial_constraints):
    boundary_edges = []
    internal_edges = []
    boundary_cells = set()

    for dc in denial_constraints:
        attrs = {x for triple in dc for x in triple[::2]}
        t1_attrs = {a.strip("'") for a in attrs if a.startswith("t1.")}
        t2_attrs = {a.strip("'") for a in attrs if a.startswith("t2.")}

        if len(t1_attrs) > 1:
            internal_edges.append(t1_attrs)

        if t1_attrs and t2_attrs:
            boundary_edges.append(attrs)
            boundary_cells.update(t1_attrs)

    return boundary_edges, internal_edges, boundary_cells

def build_rules_from_hyperedges(weighted_hypergraph_dict):
    """
    Convert hyperedges to rules for forward chaining (Algorithm 1 style).
    Each hyperedge {A, B, C} creates rules: for each cell as head, rest as tail.
    Returns: rules_map[cell] = [(tail_set, head_cell, weight), ...]
    """
    rules_map = {}
    
    for hyperedge, weight in weighted_hypergraph_dict.items():
        edge_list = list(hyperedge)
        # For each cell in the edge as head, create a rule with rest as tail
        for head_cell in edge_list:
            tail_set = frozenset(edge_list) - {head_cell}
            if head_cell not in rules_map:
                rules_map[head_cell] = []
            rules_map[head_cell].append((tail_set, head_cell, weight))
    
    return rules_map


def is_subset_minimal(premise_set, all_explanations):
    """Check if premise_set is subset-minimal (no proper subset is an explanation)"""
    premise_frozen = frozenset(premise_set)
    for exp_nodes, _, _ in all_explanations:
        if frozenset(exp_nodes) < premise_frozen:  # proper subset
            return False
    return True


def find_all_weighted_explanations(weighted_hypergraph_dict, target_node, boundary_nodes,
                                            theta):
    """
    Algorithm 1: Forward chaining enumeration with weights preserved.
    Implements: EnumerateExplanations(H_u^(θ), c_t, B_u, depth ≤ θ)
    
    For each hyperedge (set of cells), we form rules: for any cell as head,
    the other cells form the tail. When at cell x, we find rules where x is in tail.
    """
    boundary_set = set(boundary_nodes)
    
    # Q = queue of (x, P, d, weight) where:
    #   x = current cell
    #   P = accumulated premise set (union of all tail sets T(r) for rules used)
    #   d = derivation depth  
    #   weight = accumulated weight (sum of rule weights)
    queue = deque()
    explanations = []
    visited = set()  # Track visited states (x, P_frozen) to avoid cycles and duplicates
    
    # Initialize: start from target node with premise set containing target
    # Algorithm 1: Q = {(c_t, {c_t}, 0)}
    initial_P = frozenset({target_node})
    queue.append((target_node, initial_P, 0, 0.0))
    visited.add((target_node, initial_P))
    
    while queue:
        x, P, d, curr_weight = queue.popleft()
        
        # Algorithm 1 Line 4-5: If x is boundary, add explanation and continue
        # BUT: allow exploration from target even if it's a boundary (only stop after exploring)
        is_target_at_start = (d == 0 and P == {target_node} and x == target_node)
        
        if x in boundary_set and not is_target_at_start:
            # Add subset-minimal P to explanations
            explanations.append((set(P), curr_weight, d))
            continue  # Don't explore further from boundary cells
        
        # Algorithm 1 Line 6: If d < θ, explore rules
        if d < theta:
            # Algorithm 1 Line 7: For each rule r where x ∈ T(r)
            # For hyperedges: when at cell x, find hyperedges containing x
            # These represent rules where x can be part of the premises
            for hyperedge, weight in weighted_hypergraph_dict.items():
                if x in hyperedge:
                    # For this hyperedge, create a rule where:
                    # T(r) = hyperedge (all cells in the edge are premises)
                    # H(r) = other cells in hyperedge (we can derive any other cell)
                    # BUT: we can only apply if all of T(r) are in P or we're building P
                    # Actually, for forward chaining: if we're at x and x ∈ hyperedge,
                    # we can derive other cells in the hyperedge
                    other_cells = hyperedge - {x}
                    for head_cell in other_cells:
                        # Algorithm 1: new state is (h, P ∪ T(r), d+1)
                        # where T(r) = hyperedge (all cells in the edge)
                        new_P = P.union(hyperedge)  # P ∪ T(r)
                        state_key = (head_cell, frozenset(new_P))
                        
                        # Only add if we haven't visited this state before
                        if state_key not in visited:
                            visited.add(state_key)
                            new_weight = curr_weight + weight
                            queue.append((head_cell, new_P, d + 1, new_weight))
    
    # Post-process: deduplicate and ensure subset-minimal explanations
    # First deduplicate by exact node set
    unique_by_nodes = {}
    for node_set, weight, depth in explanations:
        node_frozen = frozenset(node_set)
        # Keep the one with minimum weight (or first one if weights equal)
        if node_frozen not in unique_by_nodes:
            unique_by_nodes[node_frozen] = (node_set, weight, depth)
        else:
            existing_weight = unique_by_nodes[node_frozen][1]
            if weight < existing_weight:
                unique_by_nodes[node_frozen] = (node_set, weight, depth)
    
    # Now filter for subset-minimal: remove any explanation that has a proper subset as explanation
    deduped = list(unique_by_nodes.values())
    sorted_explanations = sorted(deduped, key=lambda x: (len(x[0]), x[2]))  # by size, then depth
    
    minimal_explanations = []
    for node_set, weight, depth in sorted_explanations:
        node_frozen = frozenset(node_set)
        # Check if any existing explanation is a proper subset of this one
        has_proper_subset = False
        for existing_set, _, _ in minimal_explanations:
            if frozenset(existing_set) < node_frozen:
                has_proper_subset = True
                break
        
        if not has_proper_subset:
            # Remove any existing explanations that are proper supersets of this one
            minimal_explanations = [
                (existing_set, w, dep) for existing_set, w, dep in minimal_explanations
                if not (node_frozen < frozenset(existing_set))
            ]
            minimal_explanations.append((node_set, weight, depth))
    
    return minimal_explanations

b_edges, i_edges, b_cells = build_graph_data(dc)
#exps = find_all_weighted_explanations_weighted(set_edge_weight(i_edges), "t1.type", b_cells, 5)
