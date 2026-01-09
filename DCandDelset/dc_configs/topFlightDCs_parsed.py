# Auto-generated from flights_dc_weights_wpos_gamma0p25.csv
# Rows: 112
# Format: denial_constraints = List[List[Tuple[str,str,str]]], weights = List[float]

denial_constraints = [
    [
        ('t1.OriginWac', '!=', 't2.OriginWac'),
        ('t1.OriginStateFips', '==', 't2.OriginStateFips'),
    ],
    [
        ('t1.OriginStateFips', '!=', 't2.OriginStateFips'),
        ('t1.OriginWac', '==', 't2.OriginWac'),
    ],
    [
        ('t1.OriginAirportSeqID', '<', 't2.OriginAirportSeqID'),
        ('t1.FlightNum', '>=', 't2.FlightNum'),
        ('t1.OriginStateName', '==', 't2.OriginStateName'),
    ],
    [
        ('t1.DayofMonth', '<=', 't2.DayofMonth'),
        ('t1.OriginAirportID', '<', 't2.OriginAirportID'),
        ('t1.OriginStateName', '==', 't2.OriginStateName'),
    ],
    [
        ('t1.OriginStateName', '!=', 't2.OriginStateName'),
        ('t1.OriginCityMarketID', '==', 't2.OriginCityMarketID'),
    ],
    [
        ('t1.OriginState', '==', 't2.OriginState'),
        ('t1.OriginAirportSeqID', '<', 't2.OriginAirportSeqID'),
        ('t1.FlightNum', '>=', 't2.FlightNum'),
    ],
    [
        ('t1.OriginAirportID', '<', 't2.OriginAirportID'),
        ('t1.DayofMonth', '>=', 't2.DayofMonth'),
        ('t1.OriginStateName', '==', 't2.OriginStateName'),
    ],
    [
        ('t1.OriginCityName', '!=', 't2.OriginCityName'),
        ('t1.OriginAirportSeqID', '>=', 't2.OriginAirportSeqID'),
        ('t1.OriginAirportID', '<=', 't2.OriginAirportID'),
    ],
    [
        ('t1.OriginAirportSeqID', '==', 't2.OriginAirportSeqID'),
        ('t1.OriginState', '!=', 't2.OriginState'),
    ],
    [
        ('t1.UniqueCarrier', '==', 't2.UniqueCarrier'),
        ('t1.Carrier', '!=', 't2.Carrier'),
    ],
    [
        ('t1.Carrier', '==', 't2.Carrier'),
        ('t1.UniqueCarrier', '!=', 't2.UniqueCarrier'),
    ],
    [
        ('t1.OriginCityMarketID', '==', 't2.OriginCityMarketID'),
        ('t1.OriginState', '!=', 't2.OriginState'),
    ],
    [
        ('t1.OriginStateName', '!=', 't2.OriginStateName'),
        ('t1.Origin', '==', 't2.Origin'),
    ],
    [
        ('t1.OriginStateFips', '!=', 't2.OriginStateFips'),
        ('t1.OriginAirportSeqID', '>=', 't2.OriginAirportSeqID'),
        ('t1.OriginAirportID', '<=', 't2.OriginAirportID'),
    ],
    [
        ('t1.OriginWac', '!=', 't2.OriginWac'),
        ('t1.OriginCityName', '==', 't2.OriginCityName'),
    ],
    [
        ('t1.OriginAirportSeqID', '==', 't2.OriginAirportSeqID'),
        ('t1.OriginStateFips', '!=', 't2.OriginStateFips'),
    ],
    [
        ('t1.FlightDate', '==', 't2.FlightDate'),
        ('t1.DayOfWeek', '!=', 't2.DayOfWeek'),
    ],
    [
        ('t1.FlightNum', '>=', 't2.FlightNum'),
        ('t1.OriginWac', '==', 't2.OriginWac'),
        ('t1.OriginAirportID', '<', 't2.OriginAirportID'),
    ],
    [
        ('t1.OriginAirportSeqID', '==', 't2.OriginAirportSeqID'),
        ('t1.OriginCityName', '!=', 't2.OriginCityName'),
    ],
    [
        ('t1.OriginCityName', '==', 't2.OriginCityName'),
        ('t1.OriginAirportSeqID', '!=', 't2.OriginAirportSeqID'),
    ],
    [
        ('t1.OriginStateFips', '!=', 't2.OriginStateFips'),
        ('t1.Origin', '==', 't2.Origin'),
    ],
    [
        ('t1.AirlineID', '==', 't2.AirlineID'),
        ('t1.Carrier', '!=', 't2.Carrier'),
    ],
    [
        ('t1.AirlineID', '!=', 't2.AirlineID'),
        ('t1.Carrier', '==', 't2.Carrier'),
    ],
    [
        ('t1.OriginState', '!=', 't2.OriginState'),
        ('t1.OriginWac', '==', 't2.OriginWac'),
    ],
    [
        ('t1.OriginWac', '!=', 't2.OriginWac'),
        ('t1.OriginState', '==', 't2.OriginState'),
    ],
    [
        ('t1.OriginState', '==', 't2.OriginState'),
        ('t1.OriginAirportID', '<', 't2.OriginAirportID'),
        ('t1.DayofMonth', '>=', 't2.DayofMonth'),
    ],
    [
        ('t1.OriginState', '==', 't2.OriginState'),
        ('t1.DayofMonth', '<=', 't2.DayofMonth'),
        ('t1.OriginAirportID', '<', 't2.OriginAirportID'),
    ],
    [
        ('t1.OriginAirportSeqID', '>=', 't2.OriginAirportSeqID'),
        ('t1.OriginAirportID', '<', 't2.OriginAirportID'),
    ],
    [
        ('t1.OriginAirportSeqID', '>=', 't2.OriginAirportSeqID'),
        ('t1.OriginAirportID', '<=', 't2.OriginAirportID'),
        ('t1.OriginCityMarketID', '!=', 't2.OriginCityMarketID'),
    ],
    [
        ('t1.OriginAirportSeqID', '<', 't2.OriginAirportSeqID'),
        ('t1.FlightNum', '>=', 't2.FlightNum'),
        ('t1.OriginStateFips', '==', 't2.OriginStateFips'),
    ],
    [
        ('t1.OriginStateName', '!=', 't2.OriginStateName'),
        ('t1.OriginWac', '==', 't2.OriginWac'),
    ],
    [
        ('t1.OriginWac', '!=', 't2.OriginWac'),
        ('t1.OriginStateName', '==', 't2.OriginStateName'),
    ],
    [
        ('t1.DayOfWeek', '<=', 't2.DayOfWeek'),
        ('t1.OriginCityMarketID', '<', 't2.OriginCityMarketID'),
        ('t1.OriginStateFips', '==', 't2.OriginStateFips'),
    ],
    [
        ('t1.OriginAirportID', '==', 't2.OriginAirportID'),
        ('t1.OriginAirportSeqID', '!=', 't2.OriginAirportSeqID'),
    ],
    [
        ('t1.OriginAirportSeqID', '==', 't2.OriginAirportSeqID'),
        ('t1.OriginAirportID', '!=', 't2.OriginAirportID'),
    ],
    [
        ('t1.OriginAirportSeqID', '<', 't2.OriginAirportSeqID'),
        ('t1.OriginStateFips', '==', 't2.OriginStateFips'),
        ('t1.DayofMonth', '>=', 't2.DayofMonth'),
    ],
    [
        ('t1.DayofMonth', '<=', 't2.DayofMonth'),
        ('t1.OriginAirportSeqID', '<', 't2.OriginAirportSeqID'),
        ('t1.OriginStateFips', '==', 't2.OriginStateFips'),
    ],
    [
        ('t1.OriginStateFips', '!=', 't2.OriginStateFips'),
        ('t1.OriginCityMarketID', '==', 't2.OriginCityMarketID'),
    ],
    [
        ('t1.OriginCityMarketID', '<', 't2.OriginCityMarketID'),
        ('t1.OriginStateFips', '==', 't2.OriginStateFips'),
        ('t1.DayOfWeek', '>=', 't2.DayOfWeek'),
    ],
    [
        ('t1.OriginCityMarketID', '<', 't2.OriginCityMarketID'),
        ('t1.OriginWac', '==', 't2.OriginWac'),
        ('t1.DayOfWeek', '>=', 't2.DayOfWeek'),
    ],
    [
        ('t1.DayOfWeek', '<=', 't2.DayOfWeek'),
        ('t1.OriginCityMarketID', '<', 't2.OriginCityMarketID'),
        ('t1.OriginWac', '==', 't2.OriginWac'),
    ],
    [
        ('t1.OriginStateName', '!=', 't2.OriginStateName'),
        ('t1.OriginStateFips', '==', 't2.OriginStateFips'),
    ],
    [
        ('t1.OriginStateFips', '!=', 't2.OriginStateFips'),
        ('t1.OriginStateName', '==', 't2.OriginStateName'),
    ],
    [
        ('t1.OriginCityName', '!=', 't2.OriginCityName'),
        ('t1.Origin', '==', 't2.Origin'),
    ],
    [
        ('t1.Origin', '!=', 't2.Origin'),
        ('t1.OriginCityName', '==', 't2.OriginCityName'),
    ],
    [
        ('t1.OriginAirportSeqID', '>=', 't2.OriginAirportSeqID'),
        ('t1.OriginAirportID', '<=', 't2.OriginAirportID'),
        ('t1.OriginStateName', '!=', 't2.OriginStateName'),
    ],
    [
        ('t1.OriginState', '!=', 't2.OriginState'),
        ('t1.OriginAirportID', '==', 't2.OriginAirportID'),
    ],
    [
        ('t1.OriginWac', '!=', 't2.OriginWac'),
        ('t1.Origin', '==', 't2.Origin'),
    ],
    [
        ('t1.Origin', '!=', 't2.Origin'),
        ('t1.OriginCityMarketID', '==', 't2.OriginCityMarketID'),
    ],
    [
        ('t1.Origin', '==', 't2.Origin'),
        ('t1.OriginCityMarketID', '!=', 't2.OriginCityMarketID'),
    ],
    [
        ('t1.OriginState', '==', 't2.OriginState'),
        ('t1.OriginCityMarketID', '<', 't2.OriginCityMarketID'),
        ('t1.DayofMonth', '>=', 't2.DayofMonth'),
    ],
    [
        ('t1.OriginState', '==', 't2.OriginState'),
        ('t1.DayofMonth', '<=', 't2.DayofMonth'),
        ('t1.OriginCityMarketID', '<', 't2.OriginCityMarketID'),
    ],
    [
        ('t1.DayOfWeek', '<=', 't2.DayOfWeek'),
        ('t1.OriginState', '==', 't2.OriginState'),
        ('t1.OriginCityMarketID', '<', 't2.OriginCityMarketID'),
    ],
    [
        ('t1.OriginState', '==', 't2.OriginState'),
        ('t1.FlightNum', '>=', 't2.FlightNum'),
        ('t1.OriginAirportID', '<', 't2.OriginAirportID'),
    ],
    [
        ('t1.OriginCityMarketID', '<', 't2.OriginCityMarketID'),
        ('t1.DayOfWeek', '>=', 't2.DayOfWeek'),
        ('t1.OriginStateName', '==', 't2.OriginStateName'),
    ],
    [
        ('t1.OriginState', '==', 't2.OriginState'),
        ('t1.OriginCityMarketID', '<', 't2.OriginCityMarketID'),
        ('t1.DayOfWeek', '>=', 't2.DayOfWeek'),
    ],
    [
        ('t1.OriginWac', '!=', 't2.OriginWac'),
        ('t1.OriginAirportSeqID', '>=', 't2.OriginAirportSeqID'),
        ('t1.OriginAirportID', '<=', 't2.OriginAirportID'),
    ],
    [
        ('t1.DayOfWeek', '<=', 't2.DayOfWeek'),
        ('t1.OriginCityMarketID', '<', 't2.OriginCityMarketID'),
        ('t1.OriginStateName', '==', 't2.OriginStateName'),
    ],
    [
        ('t1.OriginStateName', '!=', 't2.OriginStateName'),
        ('t1.OriginAirportID', '==', 't2.OriginAirportID'),
    ],
    [
        ('t1.OriginWac', '!=', 't2.OriginWac'),
        ('t1.OriginAirportSeqID', '==', 't2.OriginAirportSeqID'),
    ],
    [
        ('t1.OriginState', '==', 't2.OriginState'),
        ('t1.DayofMonth', '<=', 't2.DayofMonth'),
        ('t1.OriginAirportSeqID', '<', 't2.OriginAirportSeqID'),
    ],
    [
        ('t1.OriginAirportID', '==', 't2.OriginAirportID'),
        ('t1.OriginCityMarketID', '!=', 't2.OriginCityMarketID'),
    ],
    [
        ('t1.OriginAirportID', '!=', 't2.OriginAirportID'),
        ('t1.OriginCityMarketID', '==', 't2.OriginCityMarketID'),
    ],
    [
        ('t1.OriginCityMarketID', '<', 't2.OriginCityMarketID'),
        ('t1.DayofMonth', '>=', 't2.DayofMonth'),
        ('t1.OriginStateName', '==', 't2.OriginStateName'),
    ],
    [
        ('t1.DayofMonth', '<=', 't2.DayofMonth'),
        ('t1.OriginCityMarketID', '<', 't2.OriginCityMarketID'),
        ('t1.OriginStateName', '==', 't2.OriginStateName'),
    ],
    [
        ('t1.OriginCityName', '==', 't2.OriginCityName'),
        ('t1.OriginState', '!=', 't2.OriginState'),
    ],
    [
        ('t1.OriginState', '==', 't2.OriginState'),
        ('t1.OriginAirportSeqID', '<', 't2.OriginAirportSeqID'),
        ('t1.DayofMonth', '>=', 't2.DayofMonth'),
    ],
    [
        ('t1.OriginAirportSeqID', '<', 't2.OriginAirportSeqID'),
        ('t1.FlightNum', '>=', 't2.FlightNum'),
        ('t1.OriginWac', '==', 't2.OriginWac'),
    ],
    [
        ('t1.UniqueCarrier', '!=', 't2.UniqueCarrier'),
        ('t1.AirlineID', '==', 't2.AirlineID'),
    ],
    [
        ('t1.AirlineID', '!=', 't2.AirlineID'),
        ('t1.UniqueCarrier', '==', 't2.UniqueCarrier'),
    ],
    [
        ('t1.DayofMonth', '<=', 't2.DayofMonth'),
        ('t1.OriginWac', '==', 't2.OriginWac'),
        ('t1.OriginAirportID', '<', 't2.OriginAirportID'),
    ],
    [
        ('t1.OriginWac', '!=', 't2.OriginWac'),
        ('t1.OriginCityMarketID', '==', 't2.OriginCityMarketID'),
    ],
    [
        ('t1.OriginWac', '==', 't2.OriginWac'),
        ('t1.OriginAirportID', '<', 't2.OriginAirportID'),
        ('t1.DayofMonth', '>=', 't2.DayofMonth'),
    ],
    [
        ('t1.OriginStateFips', '!=', 't2.OriginStateFips'),
        ('t1.OriginCityName', '==', 't2.OriginCityName'),
    ],
    [
        ('t1.OriginAirportSeqID', '==', 't2.OriginAirportSeqID'),
        ('t1.Origin', '!=', 't2.Origin'),
    ],
    [
        ('t1.OriginAirportSeqID', '!=', 't2.OriginAirportSeqID'),
        ('t1.Origin', '==', 't2.Origin'),
    ],
    [
        ('t1.Origin', '!=', 't2.Origin'),
        ('t1.OriginAirportID', '==', 't2.OriginAirportID'),
    ],
    [
        ('t1.OriginAirportID', '!=', 't2.OriginAirportID'),
        ('t1.Origin', '==', 't2.Origin'),
    ],
    [
        ('t1.DayofMonth', '<=', 't2.DayofMonth'),
        ('t1.OriginCityMarketID', '<', 't2.OriginCityMarketID'),
        ('t1.OriginStateFips', '==', 't2.OriginStateFips'),
    ],
    [
        ('t1.DayofMonth', '<=', 't2.DayofMonth'),
        ('t1.OriginAirportSeqID', '<', 't2.OriginAirportSeqID'),
        ('t1.OriginStateName', '==', 't2.OriginStateName'),
    ],
    [
        ('t1.OriginCityMarketID', '<', 't2.OriginCityMarketID'),
        ('t1.OriginStateFips', '==', 't2.OriginStateFips'),
        ('t1.DayofMonth', '>=', 't2.DayofMonth'),
    ],
    [
        ('t1.DayofMonth', '==', 't2.DayofMonth'),
        ('t1.DayOfWeek', '!=', 't2.DayOfWeek'),
    ],
    [
        ('t1.OriginAirportSeqID', '<', 't2.OriginAirportSeqID'),
        ('t1.DayofMonth', '>=', 't2.DayofMonth'),
        ('t1.OriginStateName', '==', 't2.OriginStateName'),
    ],
    [
        ('t1.Origin', '!=', 't2.Origin'),
        ('t1.OriginAirportSeqID', '>=', 't2.OriginAirportSeqID'),
        ('t1.OriginAirportID', '<=', 't2.OriginAirportID'),
    ],
    [
        ('t1.OriginStateName', '!=', 't2.OriginStateName'),
        ('t1.OriginCityName', '==', 't2.OriginCityName'),
    ],
    [
        ('t1.FlightNum', '>=', 't2.FlightNum'),
        ('t1.Month', '<', 't2.Month'),
    ],
    [
        ('t1.OriginStateFips', '!=', 't2.OriginStateFips'),
        ('t1.OriginAirportID', '==', 't2.OriginAirportID'),
    ],
    [
        ('t1.OriginCityName', '!=', 't2.OriginCityName'),
        ('t1.OriginCityMarketID', '==', 't2.OriginCityMarketID'),
    ],
    [
        ('t1.OriginCityName', '==', 't2.OriginCityName'),
        ('t1.OriginCityMarketID', '!=', 't2.OriginCityMarketID'),
    ],
    [
        ('t1.OriginAirportSeqID', '<', 't2.OriginAirportSeqID'),
        ('t1.OriginAirportID', '>=', 't2.OriginAirportID'),
    ],
    [
        ('t1.OriginState', '==', 't2.OriginState'),
        ('t1.OriginStateName', '!=', 't2.OriginStateName'),
    ],
    [
        ('t1.OriginState', '!=', 't2.OriginState'),
        ('t1.OriginStateName', '==', 't2.OriginStateName'),
    ],
    [
        ('t1.DayofMonth', '<=', 't2.DayofMonth'),
        ('t1.OriginAirportSeqID', '<', 't2.OriginAirportSeqID'),
        ('t1.OriginWac', '==', 't2.OriginWac'),
    ],
    [
        ('t1.OriginCityName', '!=', 't2.OriginCityName'),
        ('t1.OriginAirportID', '==', 't2.OriginAirportID'),
    ],
    [
        ('t1.OriginAirportID', '!=', 't2.OriginAirportID'),
        ('t1.OriginCityName', '==', 't2.OriginCityName'),
    ],
    [
        ('t1.OriginAirportSeqID', '==', 't2.OriginAirportSeqID'),
        ('t1.OriginCityMarketID', '!=', 't2.OriginCityMarketID'),
    ],
    [
        ('t1.OriginCityMarketID', '==', 't2.OriginCityMarketID'),
        ('t1.OriginAirportSeqID', '!=', 't2.OriginAirportSeqID'),
    ],
    [
        ('t1.OriginAirportSeqID', '<', 't2.OriginAirportSeqID'),
        ('t1.OriginWac', '==', 't2.OriginWac'),
        ('t1.DayofMonth', '>=', 't2.DayofMonth'),
    ],
    [
        ('t1.FlightDate', '!=', 't2.FlightDate'),
        ('t1.DayofMonth', '==', 't2.DayofMonth'),
    ],
    [
        ('t1.DayofMonth', '!=', 't2.DayofMonth'),
        ('t1.FlightDate', '==', 't2.FlightDate'),
    ],
    [
        ('t1.OriginState', '!=', 't2.OriginState'),
        ('t1.OriginStateFips', '==', 't2.OriginStateFips'),
    ],
    [
        ('t1.OriginState', '==', 't2.OriginState'),
        ('t1.OriginStateFips', '!=', 't2.OriginStateFips'),
    ],
    [
        ('t1.OriginAirportSeqID', '>=', 't2.OriginAirportSeqID'),
        ('t1.OriginAirportID', '<=', 't2.OriginAirportID'),
        ('t1.OriginState', '!=', 't2.OriginState'),
    ],
    [
        ('t1.OriginAirportSeqID', '==', 't2.OriginAirportSeqID'),
        ('t1.OriginStateName', '!=', 't2.OriginStateName'),
    ],
    [
        ('t1.OriginWac', '!=', 't2.OriginWac'),
        ('t1.OriginAirportID', '==', 't2.OriginAirportID'),
    ],
    [
        ('t1.FlightNum', '>=', 't2.FlightNum'),
        ('t1.OriginStateFips', '==', 't2.OriginStateFips'),
        ('t1.OriginAirportID', '<', 't2.OriginAirportID'),
    ],
    [
        ('t1.DayofMonth', '<=', 't2.DayofMonth'),
        ('t1.OriginStateFips', '==', 't2.OriginStateFips'),
        ('t1.OriginAirportID', '<', 't2.OriginAirportID'),
    ],
    [
        ('t1.OriginStateFips', '==', 't2.OriginStateFips'),
        ('t1.OriginAirportID', '<', 't2.OriginAirportID'),
        ('t1.DayofMonth', '>=', 't2.DayofMonth'),
    ],
    [
        ('t1.OriginState', '!=', 't2.OriginState'),
        ('t1.Origin', '==', 't2.Origin'),
    ],
    [
        ('t1.FlightNum', '>=', 't2.FlightNum'),
        ('t1.OriginAirportID', '<', 't2.OriginAirportID'),
        ('t1.OriginStateName', '==', 't2.OriginStateName'),
    ],
    [
        ('t1.OriginCityMarketID', '<', 't2.OriginCityMarketID'),
        ('t1.OriginWac', '==', 't2.OriginWac'),
        ('t1.DayofMonth', '>=', 't2.DayofMonth'),
    ],
    [
        ('t1.DayofMonth', '<=', 't2.DayofMonth'),
        ('t1.OriginCityMarketID', '<', 't2.OriginCityMarketID'),
        ('t1.OriginWac', '==', 't2.OriginWac'),
    ],
]

