/**
* PlayerModalController
* @namespace fsai.profiles.controllers
*/
(function () {
  'use strict';

  angular
    .module('fsai.nba.controllers')
    .controller('PlayerModalController', PlayerModalController);

  PlayerModalController.$inject = ['$uibModalInstance', 'headers', 'rows', 'pid', 'gid'];

  /**
  * @namespace PlayerModalController
  */
  function PlayerModalController($uibModalInstance, headers, rows, pid, gid) {
    var vm = this;

    vm.close = close;

    vm.pid = pid;
    vm.gid = gid;
    vm.stat_headers = [];
    vm.stat_rows = [];
    vm.displayed_headers = [];
    vm.displayed_rows = [];

    activate(headers, rows);

    /**
    * @name activate
    * @desc Actions to be performed when this controller is instantiated
    * @memberOf fsai.nba.controllers.NBAController
    */
    function activate(headers, rows) {
      vm.stat_headers = [].concat(headers);
      vm.stat_rows = [].concat(rows);
      console.log(vm.stat_headers);
      console.log(vm.stat_rows);
      vm.displayed_headers = [].concat(vm.stat_headers);
      vm.displayed_rows = [].concat(vm.stat_rows);
    }

    function close() {
      $uibModalInstance.close();
    }

  }
})();