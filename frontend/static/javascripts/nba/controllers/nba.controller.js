/**
* ProfileController
* @namespace fsai.profiles.controllers
*/
(function () {
  'use strict';

  angular
    .module('fsai.nba.controllers')
    .controller('NBAController', NBAController);

  NBAController.$inject = ['$q', '$location', 'Snackbar', 'NBAStats'];

  /**
  * @namespace NBAController
  */
  function NBAController($q, $location, Snackbar, NBAStats) {
    var vm = this;

    vm.populateTeamStats = populateTeamStats;
    vm.populatePlayerStats = populatePlayerStats;
    vm.populateStats = populateStats;
    vm.slideInNewPage = slideInNewPage;

    vm.startdate = undefined;
    vm.enddate = undefined;
    vm.stats_type = "Team";
    
    vm.team_statstypes = ['windowed', 'opponent', 'season', 'meta'];
    vm.player_statstypes = ['windowed', 'opponent_pos', 'trend_pos', 'exponential', 'homeroadsplit', 'oppsplit', 'meta'];

    vm.itemsByPage = 20
    vm.loading = false;
    vm.loaded = false;
    vm.focused = false;

    vm.stat_headers = {};
    vm.stat_rows = {};
    vm.displayed_page = undefined;
    vm.displayed_headers = [];
    vm.displayed_rows = [];

    vm.canceler = undefined;

    activate();

    /**
    * @name activate
    * @desc Actions to be performed when this controller is instantiated
    * @memberOf fsai.nba.controllers.NBAController
    */
    function activate() {
      vm.startdate = new Date();
      vm.enddate = new Date();
      vm.loading = false;
      vm.loaded = false;
      vm.focused = false;
    }

    function linkDisplayed(key) {
      vm.displayed_headers = [].concat(vm.stat_headers[key]);
      vm.displayed_rows = [].concat(vm.stat_rows[key]);
      vm.displayed_page = key;
    }

    function slideInNewTable(key) {
      vm.loading = false;
      linkDisplayed(key);
      vm.loaded = true;
      vm.focused = true;
    }

    function slideInNewPage(key) {
      console.log(key);
      if (key == vm.displayed_page) {
        return;
      }
      vm.loading = false;
      vm.loaded = true;
      vm.focused = false;
      linkDisplayed(key);
      vm.focused = true;
    }

    function slideInLoading() {
      vm.loading = true;
      vm.loaded = false;
      vm.focused = false;
    }

    function populateStats() {
      if (vm.canceler) {
        vm.canceler.resolve();
        console.log("previous request canceled");
      }
      slideInLoading();
      if (vm.stats_type == 'Player') {
        populatePlayerStats();
      } else if (vm.stats_type == 'Team') {
        populateTeamStats();
      } else {
        console.log("INVALID STATS TYPE");
      }
    }

    function populatePlayerStats() {
      vm.canceler = $q.defer();
      NBAStats.getPlayerStats(vm.startdate, vm.enddate, vm.player_statstypes, vm.canceler)
        .success(function(response, status, headers, config){
          for (var key in response) {
            vm.stat_headers[key] = response[key]['headers'];
            vm.stat_rows[key] = response[key]['rows'];
          }
        })
        .error(function(response, status, headers, config) {
          console.log("Window team stats api error");
        })
      slideInNewTable('windowed');
    }

    function populateTeamStats() {
      vm.canceler = $q.defer();
      NBAStats.getTeamStats(vm.startdate, vm.enddate, vm.team_statstypes, vm.canceler)
        .success(function(response, status, headers, config){
          for (var key in response) {
            vm.stat_headers[key] = response[key]['headers'];
            vm.stat_rows[key] = response[key]['rows'];
          }
        })
        .error(function(response, status, headers, config) {
          console.log("team stats api error");
        })
      slideInNewTable('windowed');
    }
  }
})();