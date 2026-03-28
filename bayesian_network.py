class BayesianInference:
    @staticmethod
    def infer(s):
        """
        Mimics the MAP (Maximum A Posteriori) inference of the Bayesian Network
        from the abstract state s = (s1, s2, s3, s4, s5, s6).
        
        Returns:
            Nfloor: Expected floors traveled to reach the hall call.
            Npeople: Expected number of passengers in the car when it reaches the hall call.
            Nstop: Expected number of stops before it reaches the hall call.
        """
        s1, s2, s3, s4, s5, s6 = s
        
        # Expected distance in floors based on s3 discretization
        dest_map = {1: 2.0, 2: 5.0, 3: 8.0}
        base_dist = dest_map.get(s3, 5.0)
        
        # Calculate Nfloor
        Nfloor = base_dist
        if s1 == 1 and s2 == 1: # Going up, but call is below
            Nfloor += 6.0 # Penalty for turning around
        elif s1 == 2 and s2 == 2: # Going down, but call is above
            Nfloor += 6.0
        elif s1 == 3 or s1 == 4: # Already committed to turning around
            Nfloor += 4.0
            
        # Calculate Nstop
        # Proportional to the distance to travel and the number of currently assigned tasks
        traffic_factor = base_dist / 10.0
        expected_hall_stops = s5 * traffic_factor
        expected_car_stops = s6 * traffic_factor
        Nstop = expected_hall_stops + expected_car_stops
        
        # Calculate Npeople
        # Number of passengers currently, plus expected boardings minus alightings.
        # Down-peak: most people want to go down. 
        # We roughly approximate this as current passengers + some new ones at each stop.
        Npeople = s6 + (expected_hall_stops * 1.5) - (expected_car_stops * 1.0)
        Npeople = max(0.0, min(10.0, Npeople)) # Capacity limit
        
        return Nfloor, Npeople, Nstop
