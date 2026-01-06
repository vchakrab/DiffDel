# # denial_constraints = [
# #     [('t1.measurecode', '!=', 't2.measurecode'), ('t1.measurename', '==', 't2.measurename')],
# #     [('t1.measurecode', '==', 't2.measurecode'), ('t1.measurename', '!=', 't2.measurename')],
# #     [('t1.providernumber', '==', 't2.providernumber'), ('t1.phonenumber', '!=', 't2.phonenumber')],
# #     [('t1.providernumber', '==', 't2.providernumber'), ('t1.hospitalname', '!=', 't2.hospitalname')],
# #     [('t1.zipcode', '!=', 't2.zipcode'), ('t1.phonenumber', '==', 't2.phonenumber')],
# #     [('t1.providernumber', '==', 't2.providernumber'), ('t1.zipcode', '!=', 't2.zipcode')],
# #     [('t1.measurecode', '!=', 't2.measurecode'), ('t1.stateavg', '==', 't2.stateavg')],
# #     [('t1.measurename', '!=', 't2.measurename'), ('t1.stateavg', '==', 't2.stateavg')],
# #     [('t1.city', '!=', 't2.city'), ('t1.zipcode', '==', 't2.zipcode')],
# #     [('t1.city', '!=', 't2.city'), ('t1.phonenumber', '==', 't2.phonenumber')],
# #     [('t1.providernumber', '==', 't2.providernumber'), ('t1.city', '!=', 't2.city')],
# #     [('t1.providernumber', '==', 't2.providernumber'), ('t1.countyname', '!=', 't2.countyname')],
# #     [('t1.providernumber', '==', 't2.providernumber'), ('t1.hospitalowner', '!=', 't2.hospitalowner')],
# #     [('t1.state', '==', 't2.state'), ('t1.measurename', '==', 't2.measurename'), ('t1.stateavg', '!=', 't2.stateavg')],
# #     [('t1.state', '==', 't2.state'), ('t1.measurecode', '==', 't2.measurecode'), ('t1.stateavg', '!=', 't2.stateavg')],
# #     [('t1.condition', '!=', 't2.condition'), ('t1.measurecode', '==', 't2.measurecode')],
# #     [('t1.condition', '!=', 't2.condition'), ('t1.measurename', '==', 't2.measurename')],
# #     [('t1.condition', '!=', 't2.condition'), ('t1.stateavg', '==', 't2.stateavg')],
# #     [('t1.providernumber', '==', 't2.providernumber'), ('t1.measurecode', '==', 't2.measurecode')],
# #     [('t1.providernumber', '==', 't2.providernumber'), ('t1.measurename', '==', 't2.measurename')],
# #     [('t1.providernumber', '==', 't2.providernumber'), ('t1.stateavg', '==', 't2.stateavg')],
# #     [('t1.providernumber', '==', 't2.providernumber'), ('t1.emergencyservice', '!=', 't2.emergencyservice')],
# #     [('t1.zipcode', '==', 't2.zipcode'), ('t1.countyname', '!=', 't2.countyname'), ('t1.phonenumber', '!=', 't2.phonenumber'), ('t1.hospitalowner', '==', 't2.hospitalowner')],
# #     [('t1.hospitalname', '==', 't2.hospitalname'), ('t1.state', '==', 't2.state'), ('t1.hospitalowner', '!=', 't2.hospitalowner'), ('t1.emergencyservice', '!=', 't2.emergencyservice')],
# #     [('t1.providernumber', '!=', 't2.providernumber'), ('t1.hospitalname', '==', 't2.hospitalname'), ('t1.city', '==', 't2.city'), ('t1.state', '==', 't2.state')],
# #     [('t1.hospitalname', '==', 't2.hospitalname'), ('t1.city', '==', 't2.city'), ('t1.state', '==', 't2.state'), ('t1.phonenumber', '!=', 't2.phonenumber')],
# #     [('t1.hospitalname', '==', 't2.hospitalname'), ('t1.city', '==', 't2.city'), ('t1.state', '==', 't2.state'), ('t1.zipcode', '!=', 't2.zipcode')],
# #     [('t1.zipcode', '==', 't2.zipcode'), ('t1.measurecode', '==', 't2.measurecode'), ('t1.stateavg', '!=', 't2.stateavg')],
# #     [('t1.zipcode', '==', 't2.zipcode'), ('t1.measurename', '==', 't2.measurename'), ('t1.stateavg', '!=', 't2.stateavg')],
# #     [('t1.phonenumber', '==', 't2.phonenumber'), ('t1.measurecode', '==', 't2.measurecode'), ('t1.stateavg', '!=', 't2.stateavg')],
# #     [('t1.phonenumber', '==', 't2.phonenumber'), ('t1.measurename', '==', 't2.measurename'), ('t1.stateavg', '!=', 't2.stateavg')],
# #     [('t1.providernumber', '!=', 't2.providernumber'), ('t1.hospitalname', '==', 't2.hospitalname'), ('t1.zipcode', '==', 't2.zipcode')],
# #     [('t1.hospitalname', '==', 't2.hospitalname'), ('t1.zipcode', '==', 't2.zipcode'), ('t1.phonenumber', '!=', 't2.phonenumber')],
# #     [('t1.providernumber', '!=', 't2.providernumber'), ('t1.hospitalname', '==', 't2.hospitalname'), ('t1.phonenumber', '==', 't2.phonenumber')],
# #     [('t1.hospitalname', '==', 't2.hospitalname'), ('t1.city', '==', 't2.city'), ('t1.state', '==', 't2.state'), ('t1.countyname', '!=', 't2.countyname')],
# #     [('t1.hospitalname', '==', 't2.hospitalname'), ('t1.zipcode', '==', 't2.zipcode'), ('t1.countyname', '!=', 't2.countyname')],
# #     [('t1.hospitalname', '==', 't2.hospitalname'), ('t1.countyname', '!=', 't2.countyname'), ('t1.phonenumber', '==', 't2.phonenumber')],
# #     [('t1.hospitalname', '==', 't2.hospitalname'), ('t1.state', '==', 't2.state'), ('t1.countyname', '==', 't2.countyname'), ('t1.hospitalowner', '!=', 't2.hospitalowner')],
# #     [('t1.hospitalname', '==', 't2.hospitalname'), ('t1.city', '==', 't2.city'), ('t1.state', '==', 't2.state'), ('t1.hospitalowner', '!=', 't2.hospitalowner')],
# #     [('t1.state', '!=', 't2.state'), ('t1.stateavg', '==', 't2.stateavg')],
# # ]
# denial_constraints = [
#     # dc_index = 2
#     [('t0.measurename', '!=', 't1.measurename'), ('t0.stateavg', '==', 't1.stateavg')],
#
#     # dc_index = 5
#     [('t0.measurecode', '!=', 't1.measurecode'), ('t0.stateavg', '==', 't1.stateavg')],
#
#     # dc_index = 6
#     [('t0.providernumber', '==', 't1.providernumber'), ('t0.city', '!=', 't1.city')],
#
#     # dc_index = 7
#     [('t0.phonenumber', '==', 't1.phonenumber'), ('t0.state', '!=', 't1.state')],
#
#     # dc_index = 13
#     [('t0.measurecode', '==', 't1.measurecode'), ('t0.measurename', '!=', 't1.measurename')],
#
#     # dc_index = 14
#     [('t0.measurename', '==', 't1.measurename'), ('t0.measurecode', '!=', 't1.measurecode')],
#
#     # dc_index = 15
#     [('t0.zipcode', '==', 't1.zipcode'), ('t0.city', '!=', 't1.city')],
#
#     # dc_index = 16
#     [('t0.stateavg', '!=', 't1.stateavg'),
#      ('t0.state', '==', 't1.state'),
#      ('t0.measurename', '==', 't1.measurename')],
#
#     # dc_index = 19
#     [('t0.condition', '!=', 't1.condition'), ('t0.stateavg', '==', 't1.stateavg')],
#
#     # dc_index = 20
#     [('t0.zipcode', '!=', 't1.zipcode'), ('t0.providernumber', '==', 't1.providernumber')],
#
#     # dc_index = 23
#     [('t0.stateavg', '!=', 't1.stateavg'),
#      ('t0.state', '==', 't1.state'),
#      ('t0.measurecode', '==', 't1.measurecode')],
#
#     # dc_index = 25
#     [('t0.condition', '!=', 't1.condition'), ('t0.measurecode', '==', 't1.measurecode')],
#
#     # dc_index = 27
#     [('t0.zipcode', '!=', 't1.zipcode'), ('t0.phonenumber', '==', 't1.phonenumber')],
#
#     # dc_index = 28
#     [('t0.phonenumber', '!=', 't1.phonenumber'), ('t0.providernumber', '==', 't1.providernumber')],
#
#     # dc_index = 30
#     [('t0.state', '!=', 't1.state'), ('t0.stateavg', '==', 't1.stateavg')],
#
#     # dc_index = 33
#     [('t0.providernumber', '==', 't1.providernumber'), ('t0.countyname', '!=', 't1.countyname')],
#
#     # dc_index = 38
#     [('t0.providernumber', '==', 't1.providernumber'), ('t0.hospitalowner', '!=', 't1.hospitalowner')],
#
#     # dc_index = 41
#     [('t0.phonenumber', '==', 't1.phonenumber'), ('t0.city', '!=', 't1.city')],
#
#     # dc_index = 42
#     [('t0.zipcode', '==', 't1.zipcode'), ('t0.state', '!=', 't1.state')],
#
#     # dc_index = 43
#     [('t0.providernumber', '==', 't1.providernumber'), ('t0.state', '!=', 't1.state')],
#
#     # dc_index = 45
#     [('t0.providernumber', '==', 't1.providernumber'), ('t0.hospitalname', '!=', 't1.hospitalname')],
#
#     # dc_index = 51
#     [('t0.condition', '!=', 't1.condition'), ('t0.measurename', '==', 't1.measurename')],
# ]
denial_constraints = [
    # dc_index = 2
    [('t0.MeasureName', '!=', 't1.MeasureName'), ('t0.StateAvg', '==', 't1.StateAvg')],

    # dc_index = 5
    [('t0.MeasureCode', '!=', 't1.MeasureCode'), ('t0.StateAvg', '==', 't1.StateAvg')],

    # dc_index = 6
    [('t0.ProviderNumber', '==', 't1.ProviderNumber'), ('t0.City', '!=', 't1.City')],

    # dc_index = 7
    [('t0.PhoneNumber', '==', 't1.PhoneNumber'), ('t0.State', '!=', 't1.State')],

    # dc_index = 13
    [('t0.MeasureCode', '==', 't1.MeasureCode'), ('t0.MeasureName', '!=', 't1.MeasureName')],

    # dc_index = 14
    [('t0.MeasureName', '==', 't1.MeasureName'), ('t0.MeasureCode', '!=', 't1.MeasureCode')],

    # dc_index = 15
    [('t0.ZIPCode', '==', 't1.ZIPCode'), ('t0.City', '!=', 't1.City')],

    # dc_index = 16
    [('t0.StateAvg', '!=', 't1.StateAvg'),
     ('t0.State', '==', 't1.State'),
     ('t0.MeasureName', '==', 't1.MeasureName')],

    # dc_index = 19
    [('t0.Condition', '!=', 't1.Condition'), ('t0.StateAvg', '==', 't1.StateAvg')],

    # dc_index = 20
    [('t0.ZIPCode', '!=', 't1.ZIPCode'), ('t0.ProviderNumber', '==', 't1.ProviderNumber')],

    # dc_index = 23
    [('t0.StateAvg', '!=', 't1.StateAvg'),
     ('t0.State', '==', 't1.State'),
     ('t0.MeasureCode', '==', 't1.MeasureCode')],

    # dc_index = 25
    [('t0.Condition', '!=', 't1.Condition'), ('t0.MeasureCode', '==', 't1.MeasureCode')],

    # dc_index = 27
    [('t0.ZIPCode', '!=', 't1.ZIPCode'), ('t0.PhoneNumber', '==', 't1.PhoneNumber')],

    # dc_index = 28
    [('t0.PhoneNumber', '!=', 't1.PhoneNumber'), ('t0.ProviderNumber', '==', 't1.ProviderNumber')],

    # dc_index = 30
    [('t0.State', '!=', 't1.State'), ('t0.StateAvg', '==', 't1.StateAvg')],

    # dc_index = 33
    [('t0.ProviderNumber', '==', 't1.ProviderNumber'), ('t0.CountyName', '!=', 't1.CountyName')],

    # dc_index = 38
    [('t0.ProviderNumber', '==', 't1.ProviderNumber'), ('t0.HospitalOwner', '!=', 't1.HospitalOwner')],

    # dc_index = 41
    [('t0.PhoneNumber', '==', 't1.PhoneNumber'), ('t0.City', '!=', 't1.City')],

    # dc_index = 42
    [('t0.ZIPCode', '==', 't1.ZIPCode'), ('t0.State', '!=', 't1.State')],

    # dc_index = 43
    [('t0.ProviderNumber', '==', 't1.ProviderNumber'), ('t0.State', '!=', 't1.State')],

    # dc_index = 45
    [('t0.ProviderNumber', '==', 't1.ProviderNumber'), ('t0.HospitalName', '!=', 't1.HospitalName')],

    # dc_index = 51
    [('t0.Condition', '!=', 't1.Condition'), ('t0.MeasureName', '==', 't1.MeasureName')],
]
