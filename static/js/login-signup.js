const formContainer = document.querySelector('.form-container-main');
const showSignupBtn = document.getElementById('show-signup-btn');
const showLoginBtn = document.getElementById('show-login-btn');

showSignupBtn?.addEventListener('click', (e) => {
    e.preventDefault();
    formContainer.classList.add('active-slide');
});

showLoginBtn?.addEventListener('click', (e) => {
    e.preventDefault();
    formContainer.classList.remove('active-slide');
});

const profilePictureInput = document.getElementById('profile_picture');
const imagePreview = document.getElementById('image-preview');

function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(String(email).toLowerCase());
}

function showError(input, message) {
    const form = input.closest("form");
    const errorBox = form.querySelector(".form-errors");

    if (errorBox) {
        errorBox.innerHTML = message;
    }
    input.focus();
}

if (profilePictureInput && imagePreview) {
    profilePictureInput.addEventListener('change', function (event) {
        const file = event.target.files[0];
        const form = profilePictureInput.closest("form");
        const errorBox = form?.querySelector(".form-errors");

        if (file) {
            const validTypes = ["image/jpeg", "image/png", "image/jpg", "image/gif"];
            if (!validTypes.includes(file.type)) {
                if (errorBox) errorBox.innerHTML = "Only JPG, JPEG, PNG, or GIF files are allowed.";
                profilePictureInput.value = "";
                return;
            }

            const maxSize = 2 * 1024 * 1024;
            if (file.size > maxSize) {
                if (errorBox) errorBox.innerHTML = "File size must be less than 2MB.";
                profilePictureInput.value = "";
                return;
            }

            const reader = new FileReader();
            reader.onload = function (e) {
                imagePreview.src = e.target.result;
                if (errorBox) errorBox.innerHTML = "";
            };
            reader.readAsDataURL(file);
        }
    });
}

const loginForm = document.getElementById('login-form');
if (loginForm) {
    loginForm.addEventListener('submit', function (e) {
        const email = loginForm.querySelector("input[name='email']");
        const password = loginForm.querySelector("input[name='password']");
        const errorBox = loginForm.querySelector(".form-errors");

        errorBox.innerHTML = "";

        if (!email.value.trim()) {
            e.preventDefault();
            return showError(email, "Email is required.");
        }
        if (!validateEmail(email.value)) {
            e.preventDefault();
            return showError(email, "Enter a valid email address.");
        }
        if (!password.value.trim()) {
            e.preventDefault();
            return showError(password, "Password is required.");
        }
        if (password.value.length < 6) {
            e.preventDefault();
            return showError(password, "Password must be at least 6 characters.");
        }
    });
}

const signupForm = document.getElementById('signup-form');
if (signupForm) {
    signupForm.addEventListener('submit', function (e) {
        const firstName = signupForm.querySelector("input[name='first_name']");
        const lastName = signupForm.querySelector("input[name='last_name']");
        const email = signupForm.querySelector("input[name='email']");
        const password = signupForm.querySelector("input[name='password']");
        const role = signupForm.querySelector("input[name='role']");
        const address = signupForm.querySelector("input[name='address']");
        const profilePic = signupForm.querySelector("input[name='profile_picture']");
        const errorBox = signupForm.querySelector(".form-errors");

        errorBox.innerHTML = "";

        if (!profilePic.value) {
            e.preventDefault();
            return showError(profilePic, "Profile picture is required.");
        }
        if (!firstName.value.trim()) {
            e.preventDefault();
            return showError(firstName, "First Name is required.");
        }
        if (!lastName.value.trim()) {
            e.preventDefault();
            return showError(lastName, "Last Name is required.");
        }
        if (!email.value.trim()) {
            e.preventDefault();
            return showError(email, "Email is required.");
        }
        if (!validateEmail(email.value)) {
            e.preventDefault();
            return showError(email, "Enter a valid email address.");
        }
        if (!password.value.trim()) {
            e.preventDefault();
            return showError(password, "Password is required.");
        }
        if (password.value.length < 6) {
            e.preventDefault();
            return showError(password, "Password must be at least 6 characters.");
        }
        if (!role.value.trim()) {
            e.preventDefault();
            return showError(role, "Role is required.");
        }
        if (!address.value.trim()) {
            e.preventDefault();
            return showError(address, "Address is required.");
        }
    });
}
