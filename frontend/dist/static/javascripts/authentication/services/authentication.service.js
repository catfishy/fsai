!function(){"use strict";function t(t,n){function e(t,e,o){function i(n,o,i,u){s.login(t,e)}function u(t,n,e,o){console.error("Registration failure!")}return n.post("/api/v1/accounts/",{username:o,password:e,email:t}).then(i,u)}function o(t,e){function o(t,n,e,o){s.setAuthenticatedAccount(t.data),window.location="/"}function i(t,n,e,o){console.error("Log in failure!")}return n.post("/api/v1/auth/login/",{email:t,password:e}).then(o,i)}function i(){return t.authenticatedAccount?JSON.parse(t.authenticatedAccount):void 0}function u(){return!!t.authenticatedAccount}function c(n){t.authenticatedAccount=JSON.stringify(n)}function a(){delete t.authenticatedAccount}function r(){function t(t,n,e,o){s.unauthenticate(),window.location="/"}function e(t,n,e,o){console.error("Log out failure!")}return n.post("/api/v1/auth/logout/").then(t,e)}var s={register:e,login:o,logout:r,getAuthenticatedAccount:i,isAuthenticated:u,setAuthenticatedAccount:c,unauthenticate:a};return s}angular.module("fsai.authentication.services").factory("Authentication",t),t.$inject=["$cookies","$http"]}();