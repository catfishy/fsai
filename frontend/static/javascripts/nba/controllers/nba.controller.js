/**
* NBAController
* @namespace fsai.profiles.controllers
*/
(function () {
  'use strict';

  angular
    .module('fsai.nba.controllers')
    .controller('NBAController', NBAController);

  NBAController.$inject = ['$q', '$location', '$uibModal','Snackbar', 'NBAStats'];

  /**
  * @namespace NBAController
  */
  function NBAController($q, $location, $uibModal, Snackbar, NBAStats) {
    var vm = this;

    vm.populateTeamStats = populateTeamStats;
    vm.populatePlayerStats = populatePlayerStats;
    vm.populateStats = populateStats;
    vm.slideInNewPage = slideInNewPage;
    vm.playerMatchup = playerMatchup;

    vm.startdate = undefined;
    vm.enddate = undefined;
    vm.stats_type = "Player";
    
    vm.itemsByPage = 100;

    vm.loading = false;
    vm.loaded = false;
    vm.focused = false;
    vm.popup = false;
    vm.displayed_page = undefined;
    vm.canceler = undefined;

    vm.response_headers = [];
    vm.response_rows = {};
    vm.stat_headers = [];
    vm.stat_rows = [];
    vm.displayed_headers = [];
    vm.displayed_rows = [];

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
      $.when(slideInLoading()).done(function () {
        vm.stat_rows = [].concat(vm.response_rows[key]);
        vm.stat_headers = [].concat(vm.response_headers);
        vm.displayed_rows = [].concat(vm.stat_rows);
        vm.displayed_headers = [].concat(vm.stat_headers);
        console.log('linking');
        console.log(vm.displayed_rows.length);
        vm.displayed_page = key;
      });

    }

    function slideInNewPage(key) {
      if (key == vm.displayed_page) {
        return;
      }
      console.log(key);
      $.when(linkDisplayed(key)).done(function () {
        console.log("done linking player stats");
        slideInLoaded();
      });
    }

    function slideInLoaded() {
      vm.loading = false;
      vm.empty = false;
      vm.loaded = true;
      vm.focused = true;
    }

    function slideInLoading() {
      vm.loaded = false;
      vm.focused = false;
      vm.empty = false;
      vm.loading = true;
    }

    function slideInEmpty() {
      clearPage();
    }

    function slideInPopup(pid, gid, headers, rows){
      var modalInstance = $uibModal.open({
        animation: true,
        controller: 'PlayerModalController',
        controllerAs: 'vm',
        templateUrl: '/static/templates/nba/playerMatchupModal.html',
        size: 'lg',
        resolve: {
          headers: function () {
            return headers;
          },
          rows: function () {
            return rows['ALL'];
          },
          pid: function () {
            return pid;
          },
          gid: function() {
            return gid;
          }
        }
      });
      console.log("player modal opened");
      modalInstance.result.then(function () {
        console.log("player modal closed");
      }, function () {
        console.log("player modal dismissed at: " + new Date());
      });
    }

    function clearPage(){
      vm.loading = false;
      vm.loaded = false;
      vm.focused = false;
      vm.empty = false;
    }

    function playerMatchup(row) {
      console.log("player matchup called");
      var pid_key = row['pid_key'];
      var gid_key = row['gid_key'];
      var tid_key = row['tid_key'];
      console.log(pid_key);
      console.log(gid_key);
      populateMatchupStats(pid_key, tid_key, gid_key);
    }

    function populateMatchupStats(pid, tid, gid) {
      if (vm.canceler) {
        vm.canceler.resolve();
        console.log("previous request canceled");
      }
      vm.canceler = $q.defer();
      NBAStats.getPlayerMatchupStats(pid, tid, gid, vm.canceler)
        .success(function(response, status, headers, config){
          var response_dict = JSON.parse(response);
          vm.canceler = undefined;
          slideInPopup(response_dict['pid'], 
                       response_dict['gid'], 
                       response_dict['headers'],
                       response_dict['rows']);
        })
        .error(function(response, status, headers, config) {
          if (status === -1) {
            console.log("stats request canceled");
          } else {
            console.log("player matchup stats api error");
          }
        })
    }

    function populateStats() {
      slideInLoading();
      console.log("Loading");
      if (vm.canceler) {
        vm.canceler.resolve();
        console.log("previous request canceled");
      }
      if (vm.stats_type == 'Player') {
        $.when(populatePlayerStats()).done(function () {
          console.log("done populating player stats");
        });
      } else if (vm.stats_type == 'Team') {
        $.when(populateTeamStats()).done(function () {
          console.log("done populating team stats");
        });
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
            vm.response_headers = response_dict['headers'];
            vm.response_rows = response_dict['rows'];
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
            vm.response_headers = response_dict['headers'];
            vm.response_rows = response_dict['rows'];
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