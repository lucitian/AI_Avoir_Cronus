const hamburger = document.querySelector('.menu-btn');
const menu = document.querySelector('.sidebar__menu');
const links = document.querySelectorAll('.sidebar-logo');

hamburger.addEventListener("click", () => {
    menu.classList.toggle("open");
    links.forEach(link => {
        link.classList.toggle("fade");
    })
})
