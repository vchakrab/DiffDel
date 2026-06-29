#!/usr/bin/env python3
denial_constraints = [
    [('t2.iso_country', '==', 't2.iso_country'), ('t2.scheduled_service', '!=', 't2.scheduled_service')],
    [('t2.type', '==', 't2.type'), ('t2.scheduled_service', '!=', 't2.scheduled_service'), ('t2.keywords', '!=', 't2.keywords')],
    [('t2.scheduled_service', '!=', 't2.scheduled_service'), ('t2.municipality', '==', 't2.municipality')],
    [('t2.iso_country', '==', 't2.iso_country'), ('t2.continent', '!=', 't2.continent')],
    [('t2.type', '==', 't2.type'), ('t2.continent', '==', 't2.continent'), ('t2.scheduled_service', '!=', 't2.scheduled_service')],
    [('t2.municipality', '==', 't2.municipality'), ('t2.type', '!=', 't2.type')],
    [('t2.type', '==', 't2.type'), ('t2.scheduled_service', '!=', 't2.scheduled_service'), ('t2.home_link', '!=', 't2.home_link')],
]

WEIGHTS = [
    0.3829903936131864,
    0.2152410740794191,
    0.1281024715056564,
    0.7373269363538364,
    0.3914363414566598,
    0.1489528073049025,
    0.2562156779261009,
]
