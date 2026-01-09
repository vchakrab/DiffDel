denial_constraints = [[('t1.continent', '==', 't2.continent'), ('t1.iso_country', '!=', 't2.iso_country')],
 [('t1.iso_country', '==', 't2.iso_country'), ('t1.scheduled_service', '!=', 't2.scheduled_service')],
 [('t1.municipality', '==', 't2.municipality'), ('t1.scheduled_service', '!=', 't2.scheduled_service')],
 [('t1.home_link', '==', 't2.home_link'), ('t1.scheduled_service', '!=', 't2.scheduled_service'), ('t1.type', '==', 't2.type')],
 [('t1.continent', '==', 't2.continent'), ('t1.scheduled_service', '!=', 't2.scheduled_service'), ('t1.type', '==', 't2.type')],
 [('t1.keywords', '==', 't2.keywords'), ('t1.scheduled_service', '!=', 't2.scheduled_service'), ('t1.type', '==', 't2.type')],
 [('t1.municipality', '==', 't2.municipality'), ('t1.type', '==', 't2.type')]]
