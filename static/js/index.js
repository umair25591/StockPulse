document.addEventListener('DOMContentLoaded', function () {
    // Animate on Scroll using Intersection Observer
    const scrollElements = document.querySelectorAll('.animate-on-scroll');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
                observer.unobserve(entry.target); // Animate only once
            }
        });
    }, {
        threshold: 0.1
    });

    scrollElements.forEach(el => {
        observer.observe(el);
    });

    // Skill Bar Animation
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