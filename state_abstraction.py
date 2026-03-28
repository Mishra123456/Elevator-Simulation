class StateExtractor:
    @staticmethod
    def get_abstract_state(elevator, hall_call_floor, hall_call_dir):
        """
        Extract the 6-dimensional abstract state s = (s1, s2, s3, s4, s5, s6)
        for a specific elevator and hall call pair.
        """
        s1 = StateExtractor._get_s1(elevator, hall_call_floor)
        s2 = StateExtractor._get_s2(elevator.current_floor, hall_call_floor)
        s3 = StateExtractor._get_s3(elevator.current_floor, hall_call_floor)
        s4 = StateExtractor._get_s4(hall_call_dir)
        s5 = StateExtractor._get_s5(elevator)
        s6 = StateExtractor._get_s6(elevator)
        
        return (s1, s2, s3, s4, s5, s6)

    @staticmethod
    def _get_s1(elevator, hc_floor):
        """
        s1: elevator motion state {stop, up, down, up->down, down->up}
        We map these to integers for easier processing:
        0: stop
        1: up
        2: down
        3: up->down
        4: down->up
        """
        if elevator.direction == 0:
            return 0  # stop
            
        # Find the max/min target floor this elevator is committed to
        targets = set(elevator.car_calls)
        for f in elevator.assigned_hall_calls.keys():
            targets.add(f)
            
        if elevator.direction == 1:
            highest_target = max(targets) if targets else elevator.current_floor
            if highest_target < hc_floor:
                # E.g., going up to 5, hall call at 8. It can just pick it up going up.
                pass # This condition doesn't make sense. If highest is 5, and HC is 8, it doesn't need to turn around to go to 8. But it won't be able to serve HC effectively if it doesn't know about it.
            # Actually, "up->down" means elevator is going up, but to serve the hall call,
            # it will have to eventually go down.
            # If the hall call is below the highest target, and we are going up, we might pick it up on the way down.
            # Let's simplify: compare current direction and relative position of hc.
            if hc_floor < elevator.current_floor:
                # Elevator going up, but hall call is below. Must eventually go down.
                return 3 # up->down
            return 1 # up
            
        elif elevator.direction == -1:
            if hc_floor > elevator.current_floor:
                # Elevator going down, but hall call is above. Must eventually go up.
                return 4 # down->up
            return 2 # down
        
        return 0

    @staticmethod
    def _get_s2(el_floor, hc_floor):
        """
        s2: relative position of elevator to hall call
        0: same
        1: above (elevator is above hall call)
        2: below (elevator is below hall call)
        """
        if el_floor == hc_floor:
            return 0
        elif el_floor > hc_floor:
            return 1
        else:
            return 2

    @staticmethod
    def _get_s3(el_floor, hc_floor):
        """
        s3: discretized floor distance
        1: 0 - 3 floors
        2: 4 - 6 floors
        3: >= 7 floors
        """
        dist = abs(el_floor - hc_floor)
        if dist <= 3:
            return 1
        elif dist <= 6:
            return 2
        else:
            return 3

    @staticmethod
    def _get_s4(hc_dir):
        """
        s4: direction of hall call
        1 for up, -1 for down
        """
        return hc_dir

    @staticmethod
    def _get_s5(elevator):
        """
        s5: number of hall calls assigned to this elevator (<=10)
        """
        count = sum(len(dirs) for dirs in elevator.assigned_hall_calls.values())
        return min(10, count)

    @staticmethod
    def _get_s6(elevator):
        """
        s6: number of passengers in car (<=10)
        """
        return min(10, len(elevator.passengers))
