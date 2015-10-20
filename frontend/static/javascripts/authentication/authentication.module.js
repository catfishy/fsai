(function () {
  'use strict';

  angular
    .module('fsai.authentication', [
      'fsai.authentication.controllers',
      'fsai.authentication.service'
    ]);

  angular
    .module('fsai.authentication.controllers', []);

  angular
    .module('fsai.authentication.service', ['ngCookies']);
})();