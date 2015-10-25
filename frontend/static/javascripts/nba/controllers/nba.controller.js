/**
* ProfileController
* @namespace fsai.profiles.controllers
*/
(function () {
  'use strict';

  angular
    .module('fsai.nba.controllers')
    .controller('NBAController', NBAController);

  NBAController.$inject = ['$location', 'Snackbar', 'NBAStats'];

  /**
  * @namespace NBAController
  */
  function NBAController($location, Snackbar, NBAStats) {
    var vm = this;

    vm.startdate = undefined;
    vm.enddate = undefined;
    vm.populateTeamStats = populateTeamStats;
    
    vm.itemsByPage = 20
    vm.loading = false;

    vm.stat_headers = [];
    vm.stat_rows = [];
    vm.stat_data = [];
    vm.displayed_headers = [].concat(vm.stat_headers);
    vm.displayed_rows = [].concat(vm.stat_rows);

    activate();

    /**
    * @name activate
    * @desc Actions to be performed when this controller is instantiated
    * @memberOf fsai.nba.controllers.NBAController
    */
    function activate() {
      vm.startdate = new Date();
      vm.enddate = new Date(new Date().getTime() + 24 * 60 * 60 * 1000);
    }

    function linkDisplayed() {
      vm.displayed_headers = [].concat(vm.stat_headers);
      vm.displayed_rows = [].concat(vm.stat_rows);
    }

    function populateTeamStats() {
      vm.loading = true;
      NBAStats.getTeamStats(vm.startdate, vm.enddate)
        .success(function(response, status, headers, config){
          var data = angular.fromJson(response);
          vm.stat_headers = [
            {'name': 'GAME_ID', 'key': 'gid'},
            {'name': 'TEAM_ID', 'key': 'tid'},
            {'name': 'PCT_AST_FGM', 'key': 'PCT_AST_FGM'},
            {'name': 'PTS_OFF_TOV', 'key': 'PTS_OFF_TOV'},
            {'name': 'DFGM', 'key': 'DFGM'},
            {'name': 'DFGA', 'key': 'DFGA'},
            {'name': 'DRBC', 'key': 'DRBC'},      
            {'name': 'NET_RATING', 'key': 'NET_RATING'},
            {'name': 'PCT_AST_2PM', 'key': 'PCT_AST_2PM'},
            {'name': 'OREB', 'key': 'OREB'},
            {'name': 'FGM', 'key': 'FGM'},
            {'name': 'OPP_TOV_PCT', 'key': 'OPP_TOV_PCT'},
            {'name': 'FGA', 'key': 'FGA'},
            {'name': 'PCT_UAST_3PM', 'key': 'PCT_UAST_3PM'},
            {'name': 'PASS', 'key': 'PASS'},
            {'name': 'DREB', 'key': 'DREB'},
            {'name': 'PCT_PTS_FB', 'key': 'PCT_PTS_FB'},
            {'name': 'PACE', 'key': 'PACE'},
            {'name': 'FG3_PCT', 'key': 'FG3_PCT'},
            {'name': 'PCT_PTS_FT', 'key': 'PCT_PTS_FT'},
            {'name': 'CFGM', 'key': 'CFGM'},
            {'name': 'ORBC', 'key': 'ORBC'},
            {'name': 'DFG_PCT', 'key': 'DFG_PCT'},
            {'name': 'PTS_2ND_CHANCE', 'key': 'PTS_2ND_CHANCE'},
            {'name': 'PCT_PTS_2PT_MR', 'key': 'PCT_PTS_2PT_MR'},
            {'name': 'CFG_PCT', 'key': 'CFG_PCT'},
            {'name': 'FTA_RATE', 'key': 'FTA_RATE'},
            {'name': 'PF', 'key': 'PF'},
            {'name': 'PTS', 'key': 'PTS'},
            {'name': 'PCT_PTS_2PT', 'key': 'PCT_PTS_2PT'},
            {'name': 'USG_PCT', 'key': 'USG_PCT'},
            {'name': 'FTA', 'key': 'FTA'},
            {'name': 'BLK', 'key': 'BLK'},
            {'name': 'PTS_PAINT', 'key': 'PTS_PAINT'},
            {'name': 'PCT_AST_3PM', 'key': 'PCT_AST_3PM'},
            {'name': 'FTM', 'key': 'FTM'},
            {'name': 'DREB_PCT', 'key': 'DREB_PCT'},
            {'name': 'CFGA', 'key': 'CFGA'},
            {'name': 'TIMES_TIED', 'key': 'TIMES_TIED'},
            {'name': 'FTAST', 'key': 'FTAST'},
            {'name': 'PCT_FGA_2PT', 'key': 'PCT_FGA_2PT'},
            {'name': 'AST_TOV', 'key': 'AST_TOV'},
            {'name': 'UFGA', 'key': 'UFGA'},
            {'name': 'TCHS', 'key': 'TCHS'},
            {'name': 'UFGM', 'key': 'UFGM'},
            {'name': 'PIE', 'key': 'PIE'},
            {'name': 'REB', 'key': 'REB'},
            {'name': 'OPP_FTA_RATE', 'key': 'OPP_FTA_RATE'},
            {'name': 'AST_PCT', 'key': 'AST_PCT'},
            {'name': 'SAST', 'key': 'SAST'},
            {'name': 'EFG_PCT', 'key': 'EFG_PCT'},
            {'name': 'PTS_FB', 'key': 'PTS_FB'},
            {'name': 'DEF_RATING', 'key': 'DEF_RATING'},
            {'name': 'AST', 'key': 'AST'},
            {'name': 'TM_TOV_PCT', 'key': 'TM_TOV_PCT'},
            {'name': 'TS_PCT', 'key': 'TS_PCT'},
            {'name': 'PLUS_MINUS', 'key': 'PLUS_MINUS'},
            {'name': 'OPP_PTS_FB', 'key': 'OPP_PTS_FB'},
            {'name': 'FT_PCT', 'key': 'FT_PCT'},
            {'name': 'FG_PCT', 'key': 'FG_PCT'},
            {'name': 'PCT_UAST_FGM', 'key': 'PCT_UAST_FGM'},
            {'name': 'AST_RATIO', 'key': 'AST_RATIO'},
            {'name': 'DIST', 'key': 'DIST'},
            {'name': 'OPP_OREB_PCT', 'key': 'OPP_OREB_PCT'},
            {'name': 'UFG_PCT', 'key': 'UFG_PCT'},
            {'name': 'OPP_PTS_PAINT', 'key': 'OPP_PTS_PAINT'},
            {'name': 'FG3A', 'key': 'FG3A'},
            {'name': 'OPP_PTS_2ND_CHANCE', 'key': 'OPP_PTS_2ND_CHANCE'},
            {'name': 'FG3M', 'key': 'FG3M'},
            {'name': 'TO', 'key': 'TO'},
            {'name': 'OPP_PTS_OFF_TOV', 'key': 'OPP_PTS_OFF_TOV'},
            {'name': 'RBC', 'key': 'RBC'},
            {'name': 'PCT_PTS_PAINT', 'key': 'PCT_PTS_PAINT'},
            {'name': 'PCT_UAST_2PM', 'key': 'PCT_UAST_2PM'},
            {'name': 'STL', 'key': 'STL'},
            {'name': 'OPP_EFG_PCT', 'key': 'OPP_EFG_PCT'},
            {'name': 'OREB_PCT', 'key': 'OREB_PCT'},
            {'name': 'REB_PCT', 'key': 'REB_PCT'},
            {'name': 'PCT_PTS_OFF_TOV', 'key': 'PCT_PTS_OFF_TOV'},
            {'name': 'PCT_PTS_3PT', 'key': 'PCT_PTS_3PT'},
            {'name': 'LEAD_CHANGES', 'key': 'LEAD_CHANGES'},
            {'name': 'PFD', 'key': 'PFD'},
            {'name': 'LARGEST_LEAD', 'key': 'LARGEST_LEAD'},
            {'name': 'BLKA', 'key': 'BLKA'},
            {'name': 'PCT_FGA_3PT', 'key': 'PCT_FGA_3PT'},
            {'name': 'OFF_RATING', 'key': 'OFF_RATING'},
            {'name': 'Left_Corner_3_Left_Side(L)_24+_ft_attempted', 'key': 'Left_Corner_3_Left_Side(L)_24+_ft_attempted'},
            {'name': 'Mid_Range_Right_Side(R)_16_24_ft_attempted', 'key': 'Mid_Range_Right_Side(R)_16_24_ft_attempted'},
            {'name': 'Mid_Range_Center(C)_16_24_ft_attempted', 'key': 'Mid_Range_Center(C)_16_24_ft_attempted'},            
            {'name': 'In_The_Paint_(Non_RA)_Center(C)_8_16_ft_percent', 'key': 'In_The_Paint_(Non_RA)_Center(C)_8_16_ft_percent'},
            {'name': 'Backcourt_Back_Court(BC)_Back_Court_Shot_percent', 'key': 'Backcourt_Back_Court(BC)_Back_Court_Shot_percent'},
            {'name': 'Mid_Range_Left_Side(L)_8_16_ft_percent', 'key': 'Mid_Range_Left_Side(L)_8_16_ft_percent'},
            {'name': 'Left_Corner_3_Left_Side(L)_24+_ft_percent', 'key': 'Left_Corner_3_Left_Side(L)_24+_ft_percent'},
            {'name': 'Above_the_Break_3_Center(C)_24+_ft_percent', 'key': 'Above_the_Break_3_Center(C)_24+_ft_percent'},
            {'name': 'Restricted_Area_Center(C)_Less_Than_8_ft_percent', 'key': 'Restricted_Area_Center(C)_Less_Than_8_ft_percent'},              
            {'name': 'Above_the_Break_3_Left_Side_Center(LC)_24+_ft_attempted', 'key': 'Above_the_Break_3_Left_Side_Center(LC)_24+_ft_attempted'},
            {'name': 'Mid_Range_Left_Side(L)_16_24_ft_attempted', 'key': 'Mid_Range_Left_Side(L)_16_24_ft_attempted'},
            {'name': 'Right_Corner_3_Right_Side(R)_24+_ft_attempted', 'key': 'Right_Corner_3_Right_Side(R)_24+_ft_attempted'},
            {'name': 'In_The_Paint_(Non_RA)_Left_Side(L)_8_16_ft_percent', 'key': 'In_The_Paint_(Non_RA)_Left_Side(L)_8_16_ft_percent'},
            {'name': 'Above_the_Break_3_Back_Court(BC)_Back_Court_Shot_percent', 'key': 'Above_the_Break_3_Back_Court(BC)_Back_Court_Shot_percent'},
            {'name': 'Above_the_Break_3_Left_Side_Center(LC)_24+_ft_percent', 'key': 'Above_the_Break_3_Left_Side_Center(LC)_24+_ft_percent'},
            {'name': 'Above_the_Break_3_Right_Side_Center(RC)_24+_ft_attempted', 'key': 'Above_the_Break_3_Right_Side_Center(RC)_24+_ft_attempted'},
            {'name': 'Mid_Range_Right_Side_Center(RC)_16_24_ft_attempted', 'key': 'Mid_Range_Right_Side_Center(RC)_16_24_ft_attempted'},
            {'name': 'In_The_Paint_(Non_RA)_Left_Side(L)_8_16_ft_attempted', 'key': 'In_The_Paint_(Non_RA)_Left_Side(L)_8_16_ft_attempted'},
            {'name': 'Mid_Range_Right_Side_Center(RC)_16_24_ft_percent', 'key': 'Mid_Range_Right_Side_Center(RC)_16_24_ft_percent'},
            {'name': 'In_The_Paint_(Non_RA)_Center(C)_Less_Than_8_ft_attempted', 'key': 'In_The_Paint_(Non_RA)_Center(C)_Less_Than_8_ft_attempted'},
            {'name': 'In_The_Paint_(Non_RA)_Center(C)_8_16_ft_attempted', 'key': 'In_The_Paint_(Non_RA)_Center(C)_8_16_ft_attempted'},
            {'name': 'Above_the_Break_3_Back_Court(BC)_Back_Court_Shot_attempted', 'key': 'Above_the_Break_3_Back_Court(BC)_Back_Court_Shot_attempted'},
            {'name': 'In_The_Paint_(Non_RA)_Right_Side(R)_8_16_ft_percent', 'key': 'In_The_Paint_(Non_RA)_Right_Side(R)_8_16_ft_percent'},
            {'name': 'Mid_Range_Left_Side_Center(LC)_16_24_ft_attempted', 'key': 'Mid_Range_Left_Side_Center(LC)_16_24_ft_attempted'},
            {'name': 'Right_Corner_3_Right_Side(R)_24+_ft_percent', 'key': 'Right_Corner_3_Right_Side(R)_24+_ft_percent'},
            {'name': 'Mid_Range_Left_Side_Center(LC)_16_24_ft_percent', 'key': 'Mid_Range_Left_Side_Center(LC)_16_24_ft_percent'},
            {'name': 'Mid_Range_Left_Side(L)_16_24_ft_percent', 'key': 'Mid_Range_Left_Side(L)_16_24_ft_percent'},
            {'name': 'Mid_Range_Right_Side(R)_8_16_ft_percent', 'key': 'Mid_Range_Right_Side(R)_8_16_ft_percent'},
            {'name': 'Mid_Range_Center(C)_8_16_ft_percent', 'key': 'Mid_Range_Center(C)_8_16_ft_percent'},
            {'name': 'Above_the_Break_3_Center(C)_24+_ft_attempted', 'key': 'Above_the_Break_3_Center(C)_24+_ft_attempted'},
            {'name': 'Mid_Range_Right_Side(R)_8_16_ft_attempted', 'key': 'Mid_Range_Right_Side(R)_8_16_ft_attempted'},
            {'name': 'Above_the_Break_3_Right_Side_Center(RC)_24+_ft_percent', 'key': 'Above_the_Break_3_Right_Side_Center(RC)_24+_ft_percent'},
            {'name': 'In_The_Paint_(Non_RA)_Center(C)_Less_Than_8_ft_percent', 'key': 'In_The_Paint_(Non_RA)_Center(C)_Less_Than_8_ft_percent'},
            {'name': 'Mid_Range_Center(C)_8_16_ft_attempted', 'key': 'Mid_Range_Center(C)_8_16_ft_attempted'},
            {'name': 'Mid_Range_Right_Side(R)_16_24_ft_percent', 'key': 'Mid_Range_Right_Side(R)_16_24_ft_percent'},
            {'name': 'Backcourt_Back_Court(BC)_Back_Court_Shot_attempted', 'key': 'Backcourt_Back_Court(BC)_Back_Court_Shot_attempted'},
            {'name': 'Mid_Range_Center(C)_16_24_ft_percent', 'key': 'Mid_Range_Center(C)_16_24_ft_percent'},
            {'name': 'Restricted_Area_Center(C)_Less_Than_8_ft_attempted', 'key': 'Restricted_Area_Center(C)_Less_Than_8_ft_attempted'},
            {'name': 'Mid_Range_Left_Side(L)_8_16_ft_attempted', 'key': 'Mid_Range_Left_Side(L)_8_16_ft_attempted'},
            {'name': 'In_The_Paint_(Non_RA)_Right_Side(R)_8_16_ft_attempted', 'key': 'In_The_Paint_(Non_RA)_Right_Side(R)_8_16_ft_attempted'},
          ];
          vm.stat_rows = [];
          $.each(data, function(gid, team_row){
            $.each(team_row, function(tid, value){
              value['gid'] = gid;
              value['tid'] = tid;
              vm.stat_rows.push(value);
            })
          });
          vm.loading = false;
          linkDisplayed();
        })
        .error(function(response, status, headers, config) {
          vm.loading = false;
          console.log("team stats api error");
        })
    }
  }
})();