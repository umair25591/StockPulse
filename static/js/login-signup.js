const formContainer = document.querySelector('.form-container-main');
const showSignupBtn = document.getElementById('show-signup-btn');
const showLoginBtn = document.getElementById('show-login-btn');

showSignupBtn.addEventListener('click', (e) => {
    e.preventDefault();
    formContainer.classList.add('active-slide');
});

showLoginBtn.addEventListener('click', (e) => {
    e.preventDefault();
    formContainer.classList.remove('active-slide');
});

const profilePictureInput = document.getElementById('profile_picture');
const imagePreview = document.getElementById('image-preview');

if (profilePictureInput && imagePreview) {
    profilePictureInput.addEventListener('change', function(event) {
        const file = event.target.files[0];

        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    });
}