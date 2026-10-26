def recommendation_message(current_queue, current_wait, future_wait, target_wait, current_counters, recommended_counters, arrival_rate, service_rate):
 singular=lambda n:f"{n} counter" if n==1 else f"{n} counters"
 if current_queue == 0 and current_wait == 0: return "Queue is currently clear. Existing capacity is sufficient."
 if recommended_counters > current_counters and future_wait > target_wait: return f"Increase active counters from {current_counters} to {recommended_counters} to keep predicted waiting time below the target."
 if recommended_counters < current_counters and future_wait <= target_wait: return f"Current demand can be handled with {singular(recommended_counters)} while remaining within target."
 if arrival_rate > service_rate: return "Demand is increasing, but current capacity remains sufficient."
 return "Queue is operating within the configured waiting-time target."
