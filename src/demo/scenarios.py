from src.queueing.simulator import calculate_service_rate, estimate_wait_minutes, get_congestion_status, recommend_counter_count, simulate_queue, evaluate_counter_counts

SCENARIOS={
"Normal Bank Morning":{"location_id":"bank","category":"bank","queue":6,"arrival_rate":.45,"service_time":3,"counters":3,"target_wait":15,"warning":15,"critical":30},
"Bank Rush":{"location_id":"bank","category":"bank","queue":24,"arrival_rate":1.7,"service_time":3,"counters":3,"target_wait":15,"warning":15,"critical":30},
"Canteen Lunch Rush":{"location_id":"central_canteen","category":"canteen","queue":32,"arrival_rate":3.1,"service_time":2,"counters":4,"target_wait":12,"warning":20,"critical":40},
"Library Evening":{"location_id":"central_library","category":"library","queue":11,"arrival_rate":.9,"service_time":2.5,"counters":2,"target_wait":10,"warning":10,"critical":20},
"Health Centre Sudden Surge":{"location_id":"health_centre","category":"healthcare","queue":18,"arrival_rate":1.35,"service_time":4,"counters":2,"target_wait":20,"warning":12,"critical":25},}

def calculate_demo(scenario, counters, intensity=1.0):
    settings=SCENARIOS[scenario]; rate=settings["arrival_rate"]*intensity
    horizons=[5,10,20,30]; projections={h:simulate_queue(settings["queue"],rate,settings["service_time"],counters,h)[-1]["queue"] for h in horizons}
    recommendation=recommend_counter_count(settings["queue"],rate,settings["service_time"],counters,settings["target_wait"],20,list(range(1,9)))
    return {"settings":settings,"arrival_rate":rate,"service_rate":calculate_service_rate(counters,settings["service_time"]),"current_wait":estimate_wait_minutes(settings["queue"],counters,settings["service_time"]),"projections":projections,"predicted_wait":estimate_wait_minutes(projections[30],counters,settings["service_time"]),"congestion":get_congestion_status(projections[20],settings["warning"],settings["critical"]),"recommendation":recommendation,"counter_comparison":evaluate_counter_counts(settings["queue"],rate,settings["service_time"],[max(1,counters-1),counters,counters+1],20)}
