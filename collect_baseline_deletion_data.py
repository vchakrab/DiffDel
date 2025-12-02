import csv
from datetime import time
from pathlib import Path
import time
import random

import baseline_deletion
airport_dc = Path("/Users/adhariya/src/DiffDel/DCandDelset/dc_configs/raw_constraints/topAirportDCs")
hospital_dc = Path("/Users/adhariya/src/DiffDel/DCandDelset/dc_configs/raw_constraints/topHospitalDCs")
ncvoter_dc = Path("/Users/adhariya/src/DiffDel/DCandDelset/dc_configs/raw_constraints/topNCVoterDCs")
tax_dc = Path("/Users/adhariya/src/DiffDel/DCandDelset/dc_configs/raw_constraints/topTaxDCs")

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
ncvoter_constraints = [
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
ncvoter_attributes = [
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

sizes = {"airport": 45000, "hospital": 50000, "tax": 1000, "ncvoter": 50000}
def collect_baseline_1_data_for_all_dbs():
    data_file_name = "baseline_deletion_1_data.csv"
    for dataset, attrs, in zip(["airport", "hospital", "tax", "ncvoter"], [airport_attributes, hospital_attributes, tax_attributes, ncvoter_attributes]):
            with open(data_file_name, mode = 'a', newline = '') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(f"-----{dataset}-----")
                writer.writerow("attribute,time,row,cells")
            for i in range(100):
                chosen_row = random.randint(0, sizes[dataset])
                chosen_attr = random.choice(attrs)
                start_time = time.time()
                cells_deleted = baseline_deletion.baseline_deletion_delete_all(chosen_attr, chosen_row, dataset, 0.8)
                end_time = time.time() - start_time
                with open(data_file_name, mode = 'a', newline = '') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(f"{chosen_attr},{end_time},{chosen_row},{cells_deleted}")
collect_baseline_1_data_for_all_dbs()
def collect_baseline_2_data_for_all_dbs():
    data_file_name = "baseline_deletion_2_data.csv"
    for dataset, attrs, in zip(["airport", "hospital", "tax", "ncvoter"], [airport_attributes, hospital_attributes, tax_attributes, ncvoter_attributes]):
            with open(data_file_name, mode = 'a', newline = '') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(f"-----{dataset}-----")
                writer.writerow("attribute,time,row,cells")
            for i in range(100):
                chosen_row = random.randint(0, sizes[dataset])
                chosen_attr = random.choice(attrs)
                start_time = time.time()
                cells_deleted = baseline_deletion.baseline_deletion_delete_1_from_constraints(chosen_attr, chosen_row, dataset, 0.8)
                end_time = time.time() - start_time
                with open(data_file_name, mode = 'a', newline = '') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(f"{chosen_attr},{end_time},{chosen_row},{cells_deleted}")
collect_baseline_2_data_for_all_dbs()



