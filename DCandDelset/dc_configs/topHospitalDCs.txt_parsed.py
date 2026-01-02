denial_constraints = [
    [('t1.HospitalType', '==', 't2.HospitalType'), ('t1.Sample', '==', 't2.Sample'), ('t1.EmergencyService', '!=', 't2.EmergencyService')],
    [('t1.Condition', '==', 't2.Condition'), ('t1.HospitalOwner', '==', 't2.HospitalOwner'), ('t1.EmergencyService', '!=', 't2.EmergencyService')],
    [('t1.Condition', '==', 't2.Condition'), ('t1.Sample', '==', 't2.Sample'), ('t1.HospitalType', '!=', 't2.HospitalType')],
    [('t1.State', '==', 't2.State'), ('t1.HospitalType', '!=', 't2.HospitalType')],
]
