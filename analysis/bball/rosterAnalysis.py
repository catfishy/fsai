"""
Module for choosing the optimal roster from the available players,
given their projected fantasy points

Input:
- players + projected fantasy pts for matchups tonight
- budget
- cost per player

"""

from collections import Counter, defaultdict
import itertools
import copy

class OptimalRoster(object):

    def __init__(self, budget, player_costs, player_pts, player_positions, player_teams, roster_spots):
        if not isinstance(player_costs,dict) or not isinstance(player_pts,dict) or not isinstance(player_positions,dict):
            raise Exception("Player costs/points/positions must be dicts")
        if player_costs.keys() != player_pts.keys() != player_pts.keys():
            raise Exception("Player costs/points/positions must have same keys")

        self.costs = player_costs
        self.budget = budget

        self.pts = player_pts
        self.positions = player_positions

        self.players_by_position = defaultdict(list)
        for k,v in self.positions.iteritems():
            self.players_by_position[v].append(k)
        sorted_by_position = {}
        for k,v in self.players_by_position.iteritems():
            sorted_by_position[k] = sorted(v, key=lambda x: self.costs[x])
        self.players_by_position = sorted_by_position

        self.player_teams = player_teams

        if not roster_spots:
            raise Exception("Must specify roster spots")
        else:
            self.spots = roster_spots

    def validateRoster(self, roster):
        raise Exception("Roster validation function not implemented")


    def buildCheapestRoster(self):
        '''
        players by position is already sorted by cheapest
        '''
        valid_roster = None
        used = defaultdict(list)
        temp_roster = []
        by_costs = {}
        # build initial roster
        for i, s in enumerate(self.spots):
            if s not in by_costs:
                # build lookup by cost
                by_cost = defaultdict(list)
                for pid in self.players_by_position[s]:
                    pid_cost = self.costs[pid]
                    by_cost[pid_cost].append(pid)
                # sort by pts
                for k,v in by_cost.iteritems():
                    by_cost[k] = sorted(v, key=lambda x: self.pts[x], reverse=True)
                by_costs[s] = by_cost
            else:
                by_cost = by_costs[s]
            for least in sorted(by_cost.keys()):
                chosen = False
                for sub_pid in by_cost[least]:
                    if sub_pid not in used[s]:
                        used[s].append(sub_pid)
                        temp_roster.append((sub_pid, self.costs[sub_pid], self.pts[sub_pid]))
                        chosen = True
                        break
                if chosen:
                    break
        while valid_roster is None:
            # check if valid
            if self.validateRoster(temp_roster):
                valid_roster = temp_roster
                break
            # choose possible changes by cheapest, then by most pts
            # get cheapest keys
            keyset = set()
            for s in set(self.spots):
                for key in by_costs[s]:
                    keyset.add(key)
            cheapest_key = min(keyset)

            # get subs that are that cost
            subs = defaultdict(list)
            for s in set(self.spots):
                if cheapest_key in by_costs[s]:
                    subs[s] = by_costs[s][cheapest_key]

            # choose sub with most points
            best_subs = []
            for s,pids in subs.iteritems():
                for pid in pids:
                    if pid not in used[s]:
                        best_subs.append((s,pid))
                        break
            best_sub = sorted(best_subs, key=lambda x: x[1], reverse=True)[0]

            # sub in best_sub switching out one with lowest price, breaking ties by lowest pts
            pos_players = [(i, self.costs[temp_roster[i]], self.pts[temp_roster[i]]) for i,s in enumerate(self.spots) if s == best_sub[0]]
            lowest_cost = min([x[1] for x in pos_players])
            lowest_cost_players = sorted([player for player in pos_players if player[1] == lowest_cost], key=lambda x:x[2])
            to_sub = lowest_cost_players[0]
            # sub them out
            sub_pid = best_sub[1]
            temp_roster[to_sub[0]] = (sub_pid, self.costs[sub_pid], self.pts[sub_pid])
        # return valid roster
        total_cost = sum([x[1] for x in valid_roster])
        total_pts = sum(x[2] for x in valid_roster)
        return (valid_roster, total_cost, total_pts)

    def constructOptimal(self):
        '''
        Necessary for projected points to not be similar, doesn't break ties intelligently
        '''
        print "budget: %s" % self.budget
        # build cheapest roster possible, splitting ties by maximizing pts
        current_roster, min_cost, current_pts = self.buildCheapestRoster()
        print "cheapest: %s" % current_roster
        optimals = {min_cost: (current_pts, current_roster)}
        # start looping
        current_cost = min_cost + 1
        while current_cost <= self.budget:
            # default best roster is the closest budget roster (no replacements)
            best_points = None
            best_roster = None
            sorted_keys = sorted(optimals.keys(), reverse=True)
            for last_cost in sorted_keys:
                if optimals[last_cost] == None:
                    continue
                last_pts, last_roster = optimals[last_cost]
                avai_subs = self.findAvailableSubs(last_roster, current_cost-last_cost)
                if len(avai_subs) == 0:
                    continue
                for sub_index, sub_pid in avai_subs:
                    # make the substitution and check if its the current best roster
                    new_roster = copy.deepcopy(last_roster)
                    new_roster[sub_index] = (sub_pid, self.costs[sub_pid], self.pts[sub_pid])
                    # validate new roster
                    valid = self.validateRoster(new_roster)
                    if not valid:
                        continue
                    # get best sub
                    new_cost = sum([x[1] for x in new_roster])
                    new_points = sum([x[2] for x in new_roster])
                    if new_cost != current_cost:
                        raise Exception("Something went wrong %s != %s" % (new_cost, current_cost))
                    if not best_roster or new_points > best_points:
                        best_points = new_points
                        best_roster = new_roster
            if best_roster:
                optimals[current_cost] = (best_points, best_roster)
            current_cost += 1
        srtd_pts = sorted([(k,v[0],v[1]) for k,v in optimals.iteritems()], key=lambda x: x[1], reverse=True)
        top_roster = srtd_pts[0]
        print "best: %s" % (top_roster,)
        return top_roster[2] # return just the roster

    def findAvailableSubs(self, roster, budget_increase):
        avai_subs = []
        for pos, pids in self.players_by_position.iteritems():
            chosen = [(i,x) for i, x in enumerate(roster) if self.spots[i] == pos]
            chosen_costs = [(i, x[1]) for i, x in chosen]
            chosen_pids = [x[0] for i, x in chosen]
            possible_replacement_costs = [(i, x + budget_increase) for i, x in chosen_costs]
            for pid in pids:
                new_cost = self.costs[pid]
                if pid in chosen_pids:
                    # already in roster
                    continue
                for i, possible_replacement_cost in possible_replacement_costs:
                    if new_cost != possible_replacement_cost:
                        continue
                    if self.pts[pid] <= roster[i][2]:
                        # actually gives us lower points, ignore it
                        continue
                    # found a replacement with the right cost + higher pts
                    avai_subs.append((i, pid))
        return avai_subs


class FanDuelOptimalRoster(OptimalRoster):

    def __init__(self, budget, player_costs, player_pts, player_positions, player_teams, roster_spots):
        super(FanDuelOptimalRoster, self).__init__(budget, player_costs, player_pts, player_positions, player_teams, roster_spots)
        
        # normalize
        self.budget = float(self.budget) / 100.0
        for k,v in self.costs.iteritems():
            self.costs[k] = float(v) / 100.0

    def validateRoster(self, roster):
        '''
        you must pick players from at least three different teams. 
        you may not pick more than four players from the same team.
        '''
        chosen_players = [p[0] for p in roster]
        chosen_teams = [self.player_teams[pid] for pid in chosen_players]
        unique_teams = set(chosen_teams)
        num_per_team = Counter(chosen_teams).values()
        # check at least 3 different teams
        if len(unique_teams) < 3:
            return False
        # check no more than 4 players per team
        if any([v > 4 for v in num_per_team]):
            return False
        return True

if __name__ == "__main__":
    roster_positions = ['pg', 'pg', 'sg', 'sg', 'sf', 'sf', 'pf', 'pf', 'c']
    salaries = {u'bairsca01': 3500, u'jacksre01': 4400, u'shumpim01': 3900, u'ellismo01': 7200, u'roberan03': 3500, u'millemi01': 3500, u'crawfja01': 5400, u'jefferi01': 3500, u'jamesle01': 10600, u'feltora01': 3500, u'paulch01': 10000, u'ledori01': 3500, u'davisgl01': 3500, u'jonesja02': 3500, u'mcderdo01': 3500, u'hawessp01': 3800, u'bonnema01': 3500, u'mooreet01': 3500, u'duranke01': 10700, u'udohek01': 3500, u'mariosh01': 3500, u'smithis01': 3500, u'villach01': 3700, u'noahjo01': 6800, u'jonesda02': 3500, u'collini01': 3800, u'poweldw01': 3500, u'jordade01': 9000, u'jonespe01': 3500, u'nowitdi01': 6500, u'harride01': 4500, u'morroan01': 3500, u'mozgoti01': 5700, u'westbru01': 10900, u'smithgr02': 3500, u'aminual01': 4500, u'perkike01': 3500, u'turkohe01': 3500, u'gasolpa01': 9700, u'snellto01': 3600, u'splitti01': 4000, u'lambje01': 3500, u'smithjr01': 5300, u'haywobr01': 3500, u'barnema02': 5000, u'ginobma01': 5100, u'brookaa01': 3600, u'duncati01': 7700, u'jerregr01': 3500, u'baynear01': 3500, u'josepco01': 3500, u'diawbo01': 3900, u'pendeje02': 3500, u'redicjj01': 4500, u'belinma01': 4200, u'greenda02': 6200, u'thomptr01': 5200, u'bareajo01': 5000, u'dellama01': 3500, u'wilcocj01': 3500, u'waitedi01': 4600, u'irvinky01': 8700, u'dunlemi02': 4200, u'rosede01': 7000, u'millspa02': 3500, u'riverau01': 3500, u'mohamna01': 3500, u'gibsota01': 5100, u'leonaka01': 7700, u'mirotni01': 3500, u'parsoch01': 6400, u'parketo01': 5400}
    budget = 60000
    player_teams = {u'bairsca01': u'Chicago Bulls', u'jacksre01': u'Oklahoma City Thunder', u'bonnema01': u'San Antonio Spurs', u'ellismo01': u'Dallas Mavericks', u'roberan03': u'Oklahoma City Thunder', u'ledori01': u'Dallas Mavericks', u'gasolpa01': u'Chicago Bulls', u'jefferi01': u'Dallas Mavericks', u'jamesle01': u'Cleveland Cavaliers', u'mohamna01': u'Chicago Bulls', u'paulch01': u'Los Angeles Clippers', u'davisgl01': u'Los Angeles Clippers', u'jonesja02': u'Cleveland Cavaliers', u'poweldw01': u'Dallas Mavericks', u'redicjj01': u'Los Angeles Clippers', u'mooreet01': u'Chicago Bulls', u'duranke01': u'Oklahoma City Thunder', u'smithis01': u'Oklahoma City Thunder', u'villach01': u'Dallas Mavericks', u'thomptr01': u'Cleveland Cavaliers', u'millemi01': u'Cleveland Cavaliers', u'parsoch01': u'Dallas Mavericks', u'jonesda02': u'Los Angeles Clippers', u'collini01': u'Oklahoma City Thunder', u'mozgoti01': u'Cleveland Cavaliers', u'jonespe01': u'Oklahoma City Thunder', u'haywobr01': u'Cleveland Cavaliers', u'nowitdi01': u'Dallas Mavericks', u'harride01': u'Dallas Mavericks', u'morroan01': u'Oklahoma City Thunder', u'shumpim01': u'Cleveland Cavaliers', u'westbru01': u'Oklahoma City Thunder', u'smithjr01': u'Cleveland Cavaliers', u'smithgr02': u'Dallas Mavericks', u'wilcocj01': u'Los Angeles Clippers', u'perkike01': u'Oklahoma City Thunder', u'turkohe01': u'Los Angeles Clippers', u'ginobma01': u'San Antonio Spurs', u'snellto01': u'Chicago Bulls', u'splitti01': u'San Antonio Spurs', u'lambje01': u'Oklahoma City Thunder', u'waitedi01': u'Oklahoma City Thunder', u'mariosh01': u'Cleveland Cavaliers', u'barnema02': u'Los Angeles Clippers', u'udohek01': u'Los Angeles Clippers', u'greenda02': u'San Antonio Spurs', u'duncati01': u'San Antonio Spurs', u'jerregr01': u'Oklahoma City Thunder', u'baynear01': u'San Antonio Spurs', u'josepco01': u'San Antonio Spurs', u'diawbo01': u'San Antonio Spurs', u'pendeje02': u'San Antonio Spurs', u'hawessp01': u'Los Angeles Clippers', u'belinma01': u'San Antonio Spurs', u'brookaa01': u'Chicago Bulls', u'jordade01': u'Los Angeles Clippers', u'bareajo01': u'Dallas Mavericks', u'dellama01': u'Cleveland Cavaliers', u'crawfja01': u'Los Angeles Clippers', u'irvinky01': u'Cleveland Cavaliers', u'dunlemi02': u'Chicago Bulls', u'rosede01': u'Chicago Bulls', u'millspa02': u'San Antonio Spurs', u'riverau01': u'Los Angeles Clippers', u'feltora01': u'Dallas Mavericks', u'aminual01': u'Dallas Mavericks', u'mcderdo01': u'Chicago Bulls', u'leonaka01': u'San Antonio Spurs', u'mirotni01': u'Chicago Bulls', u'gibsota01': u'Chicago Bulls', u'noahjo01': u'Chicago Bulls', u'parketo01': u'San Antonio Spurs'}
    player_positions = {u'bairsca01': 'pf', u'jacksre01': 'pg', u'shumpim01': 'sg', u'ellismo01': 'sg', u'roberan03': 'sg', u'millemi01': 'sf', u'crawfja01': 'sg', u'jefferi01': 'sf', u'jamesle01': 'sf', u'feltora01': 'pg', u'paulch01': 'pg', u'ledori01': 'sg', u'davisgl01': 'pf', u'jonesja02': 'sf', u'mcderdo01': 'sf', u'hawessp01': 'c', u'bonnema01': 'c', u'mooreet01': 'pg', u'duranke01': 'sf', u'udohek01': 'c', u'mariosh01': 'sf', u'smithis01': 'pg', u'villach01': 'pf', u'noahjo01': 'c', u'jonesda02': 'sg', u'collini01': 'pf', u'poweldw01': 'pf', u'jordade01': 'c', u'jonespe01': 'sf', u'nowitdi01': 'pf', u'harride01': 'pg', u'morroan01': 'sg', u'mozgoti01': 'c', u'westbru01': 'pg', u'smithgr02': 'pf', u'aminual01': 'sf', u'perkike01': 'c', u'turkohe01': 'sf', u'gasolpa01': 'pf', u'snellto01': 'sg', u'splitti01': 'c', u'lambje01': 'sf', u'smithjr01': 'sg', u'haywobr01': 'c', u'barnema02': 'sf', u'ginobma01': 'sg', u'brookaa01': 'pg', u'duncati01': 'pf', u'jerregr01': 'pf', u'baynear01': 'c', u'josepco01': 'pg', u'diawbo01': 'pf', u'pendeje02': 'pf', u'redicjj01': 'sg', u'belinma01': 'sg', u'greenda02': 'sg', u'thomptr01': 'pf', u'bareajo01': 'pg', u'dellama01': 'pg', u'wilcocj01': 'sg', u'waitedi01': 'sg', u'irvinky01': 'pg', u'dunlemi02': 'sf', u'rosede01': 'pg', u'millspa02': 'pg', u'riverau01': 'sg', u'mohamna01': 'c', u'gibsota01': 'pf', u'leonaka01': 'sf', u'mirotni01': 'pf', u'parsoch01': 'sf', u'parketo01': 'pg'}
    point_projections = {u'bairsca01': 3.23925, u'jacksre01': 22.158500000000004, u'bonnema01': 8.276250000000001, u'smithis01': 5.8315, u'roberan03': 9.868500000000001, u'millemi01': 3.8699999999999997, u'jefferi01': 15.00325, u'jamesle01': 45.002, u'mohamna01': 5.9094999999999995, u'crawfja01': 21.451, u'paulch01': 39.644499999999994, u'davisgl01': 4.418749999999999, u'jonesja02': 4.05275, u'mcderdo01': 3.9265000000000003, u'redicjj01': 16.581, u'mooreet01': 6.391, u'duranke01': 39.7915, u'ellismo01': 35.671749999999996, u'villach01': 14.184750000000001, u'thomptr01': 22.81875, u'collini01': 8.35825, u'parsoch01': 24.80225, u'jonesda02': 2.0340000000000003, u'ledori01': 0.041000000000000036, u'riverau01': 16.52925, u'poweldw01': 5.315250000000001, u'jonespe01': 8.31275, u'haywobr01': 4.80575, u'nowitdi01': 29.383000000000003, u'harride01': 16.464, u'morroan01': 13.668249999999999, u'mozgoti01': 23.338, u'shumpim01': 13.95825, u'westbru01': 48.108, u'smithgr02': 6.900499999999999, u'perkike01': 11.65775, u'turkohe01': 5.9295, u'gasolpa01': 39.955, u'splitti01': 16.34425, u'lambje01': 7.21825, u'wilcocj01': 2.245, u'mariosh01': 11.7645, u'barnema02': 20.0045, u'udohek01': 4.132, u'brookaa01': 16.05775, u'duncati01': 27.4725, u'jerregr01': 3.582, u'josepco01': 12.53675, u'diawbo01': 20.53225, u'pendeje02': 7.785, u'hawessp01': 18.548000000000002, u'belinma01': 12.132749999999998, u'greenda02': 26.46325, u'jordade01': 37.76325, u'bareajo01': 13.40475, u'waitedi01': 18.652749999999997, u'dellama01': 12.846999999999998, u'smithjr01': 22.136000000000003, u'ginobma01': 25.86525, u'irvinky01': 33.061499999999995, u'baynear01': 7.724749999999999, u'dunlemi02': 18.00275, u'rosede01': 28.6145, u'millspa02': 17.418, u'snellto01': 11.31125, u'feltora01': 5.88275, u'aminual01': 16.67525, u'leonaka01': 33.792500000000004, u'mirotni01': 11.697, u'gibsota01': 22.95275, u'noahjo01': 27.093500000000006, u'parketo01': 21.714}


    optros = FanDuelOptimalRoster(budget,salaries,point_projections,player_positions,player_teams,roster_positions)
    optimal = optros.constructOptimal()


