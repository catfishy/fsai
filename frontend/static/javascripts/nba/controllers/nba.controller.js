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
    vm.stats_type = "Player";
    
    vm.itemsByPage = 100;

    vm.loading = false;
    vm.loaded = false;
    vm.focused = false;

    vm.stat_headers = [];
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
      vm.displayed_headers = [].concat(vm.stat_headers);
      vm.displayed_rows = [].concat(vm.stat_rows[key]);
      vm.displayed_page = key;
    }

    function clearPage(){
      vm.loading = false;
      vm.loaded = false;
      vm.focused = false;
      vm.empty = false;
    }

    function slideInNewTable(key) {
      linkDisplayed(key);
      slideInLoaded();
    }

    function slideInNewPage(key) {
      console.log(key);
      if (key == vm.displayed_page) {
        return;
      }
      slideInLoading();
      linkDisplayed(key);
      slideInLoaded();

    }

    function slideInLoaded() {
      vm.loading = false;
      vm.loaded = true;
      vm.focused = true;
      vm.empty = false;
    }

    function slideInLoading() {
      vm.loading = true;
      vm.loaded = false;
      vm.focused = false;
      vm.empty = false;
    }

    function slideInEmpty() {
      clearPage();
    }

    function populateStats() {
      if (vm.canceler) {
        vm.canceler.resolve();
        console.log("previous request canceled");
      }
      slideInLoading();
      console.log("Loading");
      if (vm.stats_type == 'Player') {
        populatePlayerStats();
      } else if (vm.stats_type == 'Team') {
        populateTeamStats();
      } else {
        console.log("INVALID STATS TYPE");
        clearPage();
      }
    }

    function populatePlayerStats() {
      vm.canceler = $q.defer();
      NBAStats.getPlayerStats(vm.startdate, vm.enddate, vm.canceler)
        .success(function(response, status, headers, config){
          var response_dict = JSON.parse(response);
          if (response_dict['rows'].length == 0) {
            slideInEmpty();
          } else {
            vm.stat_headers = response_dict['headers'];
            vm.stat_rows = response_dict['rows'];
            slideInNewPage('G');
          }
          vm.canceler = undefined;
        })
        .error(function(response, status, headers, config) {
          if (status === -1) {
            console.log("stats request canceled");
          } else {
            console.log("player stats api error");
            clearPage();
          }
        })
    }

    function populateTeamStats() {
      vm.canceler = $q.defer();
      NBAStats.getTeamStats(vm.startdate, vm.enddate, vm.canceler)
        .success(function(response, status, headers, config){
          var response_dict = JSON.parse(response);
          if (response_dict['rows'].length == 0) {
            slideInEmpty();
          } else {
            vm.stat_headers = response_dict['headers'];
            vm.stat_rows = response_dict['rows'];
            slideInNewPage('ALL');
          }
          vm.canceler = undefined;
        })
        .error(function(response, status, headers, config) {
          if (status === -1) {
            console.log("stats request canceled");
          } else {
            console.log("player stats api error");
            clearPage();
          }
        })
    }
  }
})();