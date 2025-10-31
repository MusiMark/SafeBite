// Theme Toggle
const themeToggle = document.getElementById('themeToggle');
const currentTheme = localStorage.getItem('theme') || 'light';

document.documentElement.setAttribute('data-theme', currentTheme);

if (themeToggle) {
  themeToggle.addEventListener('click', () => {
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
  });
}

// Mobile Menu
const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
const navMenu = document.querySelector('.nav-menu');

if (mobileMenuBtn && navMenu) {
  mobileMenuBtn.addEventListener('click', () => {
    navMenu.classList.toggle('active');
  });
}

// Smooth Scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  });
});

// Form Handling
document.querySelectorAll('form').forEach(form => {
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(form);
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    
    // Show loading state
    submitBtn.textContent = 'Processing...';
    submitBtn.disabled = true;
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Show success message
      const successMsg = document.createElement('div');
      successMsg.className = 'alert success';
      successMsg.textContent = 'Form submitted successfully!';
      form.appendChild(successMsg);
      
      // Reset form
      form.reset();
      
      setTimeout(() => {
        successMsg.remove();
      }, 5000);
      
    } catch (error) {
      console.error('Form submission error:', error);
    } finally {
      submitBtn.textContent = originalText;
      submitBtn.disabled = false;
    }
  });
});

// Initialize Liquid Background
function initLiquidBackground() {
  const liquidBg = document.querySelector('.liquid-bg');
  if (!liquidBg) return;
  
  const colors = ['#22c55e', '#0ea5e9', '#8b5cf6'];
  
  colors.forEach((color, index) => {
    const blob = document.createElement('div');
    blob.className = 'liquid-blob';
    blob.style.background = color;
    blob.style.width = `${300 + index * 100}px`;
    blob.style.height = `${300 + index * 100}px`;
    blob.style.top = `${-100 + index * 50}px`;
    blob.style.left = `${-50 + index * 100}px`;
    blob.style.animationDelay = `${index * 5}s`;
    
    liquidBg.appendChild(blob);
  });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  initLiquidBackground();
  
  // Add intersection observer for animations
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('animate-in');
      }
    });
  }, { threshold: 0.1 });
  
  // Observe elements for animation
  document.querySelectorAll('.feature-card, .step, .stat').forEach(el => {
    observer.observe(el);
  });
});