"""
Version 1:

- Find out what the upcoming games are
- Find out which players are likely to play which positions by team roster, player positions, and previous minutes played
- For the stats that matter (get points), calculate trajectory for each player in last 15 games
- ? do projections somehow (incorporating matchup context)

- Grab price for each player (crawl it somehow)
    - Given a HARD projection for each player X stat category
        - start with players that give you best projected points for each position, per price point
        - eliminate players that would give you lower overall points for each position
        - if there is budget leftover, can start spending suboptimally:
            - recursively:
                - eliminate players that are too expensive (would put you over budget given your current roster)
                - taking all positions together:
                    - find the player/replaced player duo that would give you the highest improvement
                    - calculate improvement by (fantasy point difference)/(price point difference)
            - keep going until no improvement players are left

"""
from datetime import datetime, timedelta

from statsETL.bball.NBAcrawler import upcomingGameCrawler
from statsETL.db.mongolib import MongoConn, NBA_DB

# get nba db conn
nba_conn = MongoConn(db=NBA_DB)

# get collections
team_game_collection = nba_conn.getCollection("team_games")
player_game_collection = nba_conn.getCollection("player_games")
game_collection = nba_conn.getCollection("games")
team_collection = nba_conn.getCollection("teams")
player_collection = nba_conn.getCollection("players")


def upcomingGames():
    today = datetime.now() + timedelta(int(1))
    gl_crawl = upcomingGameCrawler(date=today)
    return gl_crawl.crawlPage()


def teamInfo(team_id):
    result = team_collection.find_one({"_id":str(team_id)})
    return result


def availablePlayers(upcoming_game_dict):
    away_info = teamInfo(upcoming_game_dict['away_team_id'])
    home_info = teamInfo(upcoming_game_dict['home_team_id'])
    return {"home_players": home_info['players'], "away_players": away_info['players']}


def playerRecentStats(player_id, games_back=15):
    results = player_game_collection.find({"player_id":str(player_id)}, sort=[("game_time",-1)], limit=games_back)
    return list(results)

def fanDuelFantasyPoints(player_stat_row):
    pass

def basicPlayerStatTrajectories(player_stats):
    """
    derive trajectories for MP, PTS, AST, STL, BLK, 3P, TRB
    """
    keys = ['MP', 'PTS', 'AST', 'STL', 'BLK', '3P', 'TRB']
    trajectories = {k:[] for k in keys}
    for stat in player_stats:
        if stat['MP'] == 0:
            for k in keys:
                trajectories[k].append(0.0)
            continue
        else:
            for k in keys:
                trajectories[k].append(stat[k])
    return trajectories

if __name__=="__main__":
    games = upcomingGames()
    g = games[-1]
    players = availablePlayers(g)
    # find a random player for now
    player_ids = players['away_players']
    for player_id in player_ids:
        print player_id
        recent_stats = playerRecentStats(player_id)
        trajectories = basicPlayerStatTrajectories(recent_stats)
        for k,v in trajectories.items():
            print "%s: %s" % (k,v)
        print '---'

'''
{ "_id" : ObjectId("54d0237e8c24bd3107be5213"), "STL%" : 0, "FT" : 3, 
"3P" : 2, "TOV" : 1, "FG" : 4, "player_id" : "gallola01", "3PA" : 4, 
"DRB" : 5, "ORB%" : 0, "BLK%" : 2.7, "AST" : 2, "game_time" : ISODate("2015-02-01T14:00:00Z"), 
"FT%" : 1, "3PAr" : 0.364, "PF" : 3, "PTS" : 13, "FGA" : 11, "FG%" : 0.364, "STL" : 0, "TRB" : 5, 
"TOV%" : 7.5, "AST%" : 10.6, "FTA" : 3, "eFG%" : 0.455, "BLK" : 1, "game_id" : "201502010NYK", 

"FTr" : 0.273, "+/-" : 18, "USG%" : 21.6, "DRB%" : 20.2, "TS%" : 0.528, "MP" : "29:42", 
"DRtg" : 94, "ORtg" : 108, "TRB%" : 9.3, "ORB" : 0, "3P%" : 0.5 }
'''