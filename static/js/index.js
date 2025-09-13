document.addEventListener('DOMContentLoaded', function () {
    const scrollElements = document.querySelectorAll('.animate-on-scroll');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1
    });

    scrollElements.forEach(el => {
        observer.observe(el);
    });

    const skillObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const skillCircles = entry.target.querySelectorAll('.skill-circle');
                skillCircles.forEach(circle => {
                    const targetPercent = parseInt(circle.getAttribute('data-percent'));
                    const percentSpan = circle.querySelector('.skill-percent');
                    let currentPercent = 0;

                    const animate = () => {
                        if (currentPercent < targetPercent) {
                            currentPercent++;
                            percentSpan.textContent = currentPercent + '%';
                            circle.style.setProperty('--percent', currentPercent);
                            requestAnimationFrame(animate);
                        }
                    };
                    requestAnimationFrame(animate);
                });
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    const skillsGrid = document.querySelector('.skills-grid');
    if (skillsGrid) {
        skillObserver.observe(skillsGrid);
    }
});

const testimonials = document.querySelectorAll('.testimonial-box');
const prev = document.getElementById('prev');
const next = document.getElementById('next');
let index = 0;

function showTestimonial(i) {
    testimonials.forEach((t, idx) => {
        t.classList.toggle('active', idx === i);
    });
}

next.addEventListener('click', () => {
    index = (index + 1) % testimonials.length;
    showTestimonial(index);
});

prev.addEventListener('click', () => {
    index = (index - 1 + testimonials.length) % testimonials.length;
    showTestimonial(index);
});

// Auto slide every 5s
setInterval(() => {
    index = (index + 1) % testimonials.length;
    showTestimonial(index);
}, 5000);
