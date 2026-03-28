import heapq
import random
import numpy as np

# Typical time constants (in seconds)
FLOOR_TRAVEL_TIME = 2.0
DOOR_OPEN_TIME = 2.0
DOOR_CLOSE_TIME = 2.0
BOARD_ALIGHT_TIME = 1.0

class Passenger:
    def __init__(self, id, origin, destination, arrival_time):
        self.id = id
        self.origin = origin
        self.destination = destination
        self.arrival_time = arrival_time
        self.board_time = -1
        self.alight_time = -1

    @property
    def wait_time(self):
        return (self.board_time - self.arrival_time) if self.board_time >= 0 else 0

    @property
    def travel_time(self):
        return (self.alight_time - self.board_time) if self.alight_time >= 0 else 0

class Elevator:
    def __init__(self, id, capacity=10):
        self.id = id
        self.capacity = capacity
        self.current_floor = 1
        self.direction = 0  # 1 (up), -1 (down), 0 (stationary)
        self.passengers = []  # passengers currently in the car
        
        # Assigned tasks:
        # Dictionary of floor -> set of directions for hall calls
        # e.g., {5: {1, -1}} means hall calls to go up and down at floor 5
        self.assigned_hall_calls = {} 
        self.car_calls = set() # Set of destination floors (car calls)
        
        # SMDP / Control states
        self.state = "IDLE" # IDLE, MOVING, DOORS_OPEN
        self.target_floor = None

    def add_hall_call(self, floor, direction):
        if floor not in self.assigned_hall_calls:
            self.assigned_hall_calls[floor] = set()
        self.assigned_hall_calls[floor].add(direction)

    def remove_hall_call(self, floor, direction):
        if floor in self.assigned_hall_calls and direction in self.assigned_hall_calls[floor]:
            self.assigned_hall_calls[floor].remove(direction)
            if not self.assigned_hall_calls[floor]:
                del self.assigned_hall_calls[floor]

    def add_car_call(self, floor):
        self.car_calls.add(floor)

    def remove_car_call(self, floor):
        if floor in self.car_calls:
            self.car_calls.remove(floor)

    def has_calls(self):
        return len(self.car_calls) > 0 or len(self.assigned_hall_calls) > 0

    def next_target_scan(self):
        """Standard SCAN elevator algorithm with capacity awareness."""
        if not self.has_calls():
            self.direction = 0
            return None

        # If the car is full, it should only consider car calls (destinations)
        if len(self.passengers) >= self.capacity:
            targets = set(self.car_calls)
            if not targets:
                # Should not happen if passengers are inside
                self.direction = 0
                return None
        else:
            # All floors that need to be visited
            targets = set(self.car_calls)
            for f in self.assigned_hall_calls.keys():
                targets.add(f)

        if not targets:
            self.direction = 0
            return None

        if self.direction == 0:
            if any(f > self.current_floor for f in targets):
                self.direction = 1
            elif any(f < self.current_floor for f in targets):
                self.direction = -1
            else:
                self.direction = 1 # Force a direction if target is current floor

        if self.direction == 1:
            above_targets = [f for f in targets if f >= self.current_floor]
            if above_targets:
                return min(above_targets)
            else:
                self.direction = -1
                below_targets = [f for f in targets if f <= self.current_floor]
                return max(below_targets) if below_targets else None
        else:
            below_targets = [f for f in targets if f <= self.current_floor]
            if below_targets:
                return max(below_targets)
            else:
                self.direction = 1
                above_targets = [f for f in targets if f >= self.current_floor]
                return min(above_targets) if above_targets else None

class ElevatorEnv:
    def __init__(self, num_floors=10, num_elevators=4, traffic_type='down-peak', arrival_rate=0.1):
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.traffic_type = traffic_type # 'down-peak', 'mixed'
        self.arrival_rate = arrival_rate # Passengers per second overall
        
        self.reset()

    def reset(self):
        self.elevators = [Elevator(i) for i in range(self.num_elevators)]
        self.time = 0.0
        self.eq = [] # Event queue (heap)
        self.passenger_id_counter = 0
        
        self.finished_passengers = []
        self.waiting_passengers = {} # (floor, direction) -> list of Passengers
        
        self._schedule_next_arrival()
        
        return self._get_global_state()

    def _schedule_next_arrival(self):
        # Exponential inter-arrival time
        inter_arrival = random.expovariate(self.arrival_rate)
        arr_time = self.time + inter_arrival
        
        floor, dest = self._generate_traffic_origin_dest()
        heapq.heappush(self.eq, (arr_time, 'ARRIVAL', (floor, dest)))

    def _generate_traffic_origin_dest(self):
        if self.traffic_type == 'down-peak':
            # Traffic mostly from upper floors to floor 1
            if random.random() < 0.9:
                floor = random.randint(2, self.num_floors)
                dest = 1
            else:
                floor = 1
                dest = random.randint(2, self.num_floors)
        else: # mixed
            floor = random.randint(1, self.num_floors)
            dest = random.randint(1, self.num_floors)
            while dest == floor:
               dest = random.randint(1, self.num_floors)
        return floor, dest

    def _get_global_state(self):
        # The agent uses this alongside the new call to compute abstract states
        return {
            'elevators': self.elevators,
            'waiting': self.waiting_passengers,
            'time': self.time
        }

    def _trigger_elevator_movement(self, el_id):
        el = self.elevators[el_id]
        if el.state == "IDLE":
            target = el.next_target_scan()
            if target is not None:
                if target == el.current_floor:
                    el.state = "DOORS_OPEN"
                    heapq.heappush(self.eq, (self.time + DOOR_OPEN_TIME, 'DOORS_OPENED', el_id))
                else:
                    el.state = "MOVING"
                    el.target_floor = target
                    distance = abs(el.target_floor - el.current_floor)
                    arr_time = self.time + distance * FLOOR_TRAVEL_TIME
                    heapq.heappush(self.eq, (arr_time, 'EL_ARRIVES', el_id))

    def step_assign(self, assigned_el_id, call_floor, call_direction):
        """Agent calls this to assign a recently arrived hall call"""
        el = self.elevators[assigned_el_id]
        el.add_hall_call(call_floor, call_direction)
        if el.state == "IDLE":
            self._trigger_elevator_movement(assigned_el_id)
        
        # Advance simulation to next ARRIVAL (decision epoch)
        return self._run_sim_until_next_decision()

    def _run_sim_until_next_decision(self):
        reward = 0.0
        # Process events until the next ARRIVAL
        while self.eq:
            event_time, event_type, data = heapq.heappop(self.eq)
            self.time = event_time
            
            # Trace events for debugging (uncomment for deep dive)
            # print(f"TRACE: t={self.time:.2f} | {event_type} | Data={data}")
            
            if event_type == 'ARRIVAL':
                floor, dest = data
                direction = 1 if dest > floor else -1
                
                p = Passenger(self.passenger_id_counter, floor, dest, self.time)
                self.passenger_id_counter += 1
                
                key = (floor, direction)
                if key not in self.waiting_passengers:
                    self.waiting_passengers[key] = []
                self.waiting_passengers[key].append(p)
                
                self._schedule_next_arrival()
                
                # Yield control to RL agent for assignment
                state = self._get_global_state()
                # Compute reward accumulated up to this decision epoch
                return state, p, False
                
            elif event_type == 'EL_ARRIVES':
                el_id = data
                el = self.elevators[el_id]
                el.current_floor = el.target_floor
                # It arrived, open doors
                el.state = "DOORS_OPEN"
                heapq.heappush(self.eq, (self.time + DOOR_OPEN_TIME, 'DOORS_OPENED', el_id))
                
            elif event_type == 'DOORS_OPENED':
                el_id = data
                el = self.elevators[el_id]
                
                # Handle alighting
                alighting_count = sum(1 for p in el.passengers if p.destination == el.current_floor)
                for p in [p for p in el.passengers if p.destination == el.current_floor]:
                    p.alight_time = self.time
                    self.finished_passengers.append(p)
                el.passengers = [p for p in el.passengers if p.destination != el.current_floor]
                el.remove_car_call(el.current_floor)
                
                # Handle boarding
                boarding_count = 0
                
                # Logic: Any passenger waiting at this floor whose direction is assigned to THIS elevator should board.
                # Standard SCAN would only pick up those in current direction, but if we are at the end of our current 
                # run or specifically sent here for a call, we must be able to switch.
                
                # Check what directions this elevator is assigned to at this floor
                assigned_dirs = el.assigned_hall_calls.get(el.current_floor, set()).copy()
                
                # Determine which directions to check for boarding
                boarding_keys_to_check = []
                for d in [1, -1]:
                    if d in assigned_dirs or el.direction == 0:
                        boarding_keys_to_check.append((el.current_floor, d))
                    elif d == el.direction:
                         boarding_keys_to_check.append((el.current_floor, d))

                for key in boarding_keys_to_check:
                    if key in self.waiting_passengers and len(el.passengers) < el.capacity:
                        waiting_list = self.waiting_passengers[key]
                        while waiting_list and len(el.passengers) < el.capacity:
                            p = waiting_list.pop(0)
                            p.board_time = self.time
                            el.passengers.append(p)
                            el.add_car_call(p.destination)
                            boarding_count += 1
                        
                        if not waiting_list:
                            del self.waiting_passengers[key]
                            el.remove_hall_call(el.current_floor, key[1])
                            
                # Re-evaluate el.direction if it's now out of targets in its current direction
                targets = set(el.car_calls)
                for f in el.assigned_hall_calls.keys():
                    targets.add(f)
                
                if el.direction == 1 and not any(f > el.current_floor for f in targets):
                    if any(f < el.current_floor for f in targets):
                        el.direction = -1
                    else:
                        el.direction = 0
                elif el.direction == -1 and not any(f < el.current_floor for f in targets):
                    if any(f > el.current_floor for f in targets):
                        el.direction = 1
                    else:
                        el.direction = 0

                transfer_time = (alighting_count + boarding_count) * BOARD_ALIGHT_TIME
                # Close doors after transfer
                heapq.heappush(self.eq, (self.time + transfer_time + DOOR_CLOSE_TIME, 'DOORS_CLOSED', el_id))
                
            elif event_type == 'DOORS_CLOSED':
                el_id = data
                el = self.elevators[el_id]
                el.state = "IDLE"
                self._trigger_elevator_movement(el_id)
                
        return None, None, True

    def calculate_reward(self):
        # Calculate -sqrt(sum(wait^2) + sum(travel^2)) for ALL active passengers
        # wait_time = current_time - arrival_time (if not boarded)
        # travel_time = current_time - board_time (if boarded)
        
        total_wait_sq = 0
        total_travel_sq = 0
        
        # 1. Waiting passengers
        for key, waiting_list in self.waiting_passengers.items():
            for p in waiting_list:
                wait = self.time - p.arrival_time
                total_wait_sq += wait ** 2
        
        # 2. Passengers in cars
        for el in self.elevators:
            for p in el.passengers:
                # They already waited, now they are traveling
                wait = p.board_time - p.arrival_time
                travel = self.time - p.board_time
                total_wait_sq += wait ** 2
                total_travel_sq += travel ** 2
                
        return -np.sqrt(total_wait_sq + total_travel_sq)
