!function(){"use strict";function e(e,t,o,r,n){function u(){function u(e,t,o,r){c.profile=e.data}function i(t,o,r,u){e.url("/"),n.error("That user does not exist.")}var a=o.getAuthenticatedAccount(),s=t.username.substr(1);a?a.username!==s&&(e.url("/"),n.error("You are not authorized to view this page.")):(e.url("/"),n.error("You are not authorized to view this page.")),r.get(s).then(u,i)}function i(){function e(e,t,r,u){o.unauthenticate(),window.location="/",n.show("Your account has been deleted.")}function t(e,t,o,r){n.error(e.error)}r.destroy(c.profile.username).then(e,t)}function a(){function e(e,t,o,r){n.show("Your profile has been updated.")}function t(e,t,o,r){n.error(e.error)}r.update(c.profile).then(e,t)}var c=this;c.destroy=i,c.update=a,u()}angular.module("fsai.profiles.controllers").controller("ProfileSettingsController",e),e.$inject=["$location","$routeParams","Authentication","Profile","Snackbar"]}();