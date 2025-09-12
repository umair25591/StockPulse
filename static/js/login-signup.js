const loginBox = document.querySelector('.login-box');
const showSignupLink = document.getElementById('show-signup');
const showLoginLink = document.getElementById('show-login');

showSignupLink.addEventListener('click', (e) => {
    e.preventDefault();
    loginBox.classList.add('active-slide');
});

showLoginLink.addEventListener('click', (e) => {
    e.preventDefault();
    loginBox.classList.remove('active-slide');
});