/**
 * Hotel Price Prediction System — Frontend Logic
 * ================================================
 * Handles form submission, API calls, and dynamic rendering of results.
 */

document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

/**
 * Initialize the application.
 */
function initApp() {
    loadOptions();
    bindEvents();
}

/**
 * Load dropdown options from the API.
 */
async function loadOptions() {
    try {
        const response = await fetch('/api/options');
        const data = await response.json();

        populateSelect('city', data.cities.map(c => ({
            value: c.name,
            label: `${c.name}, ${c.state}`
        })));

        populateSelect('room_type', data.room_types.map(r => ({
            value: r,
            label: r
        })));

        populateSelect('season', data.seasons.map(s => ({
            value: s,
            label: s
        })));

    } catch (error) {
        console.error('Failed to load options:', error);
    }
}

/**
 * Populate a <select> element with options.
 */
function populateSelect(id, options) {
    const select = document.getElementById(id);
    if (!select) return;

    // Keep the first placeholder option if exists
    const placeholder = select.querySelector('option[value=""]');
    select.innerHTML = '';
    if (placeholder) select.appendChild(placeholder);

    options.forEach(opt => {
        const option = document.createElement('option');
        option.value = opt.value;
        option.textContent = opt.label;
        select.appendChild(option);
    });
}

/**
 * Bind event listeners.
 */
function bindEvents() {
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', handleSubmit);
    }
}

/**
 * Handle form submission.
 */
async function handleSubmit(e) {
    e.preventDefault();

    const btn = document.getElementById('predictBtn');
    const resultsSection = document.getElementById('results');

    // Validate
    if (!validateForm()) return;

    // Set loading state
    btn.classList.add('loading');
    resultsSection.classList.remove('active');

    // Gather form data
    const payload = {
        city: document.getElementById('city').value,
        season: document.getElementById('season').value,
        num_guests: parseInt(document.getElementById('num_guests').value) || 2,
        room_type: document.getElementById('room_type').value,
        is_weekend: document.getElementById('is_weekend').checked,
        has_events: document.getElementById('has_events').checked,
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.success) {
            renderResults(data);
            resultsSection.classList.add('active');
            // Smooth scroll to results
            setTimeout(() => {
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);
        } else {
            showError(data.error || 'Prediction failed. Please try again.');
        }

    } catch (error) {
        console.error('Prediction error:', error);
        showError('Network error. Please ensure the server is running.');
    } finally {
        btn.classList.remove('loading');
    }
}

/**
 * Validate form inputs.
 */
function validateForm() {
    const city = document.getElementById('city').value;
    const season = document.getElementById('season').value;
    const roomType = document.getElementById('room_type').value;
    const guests = document.getElementById('num_guests').value;

    if (!city) {
        highlightField('city');
        return false;
    }
    if (!season) {
        highlightField('season');
        return false;
    }
    if (!roomType) {
        highlightField('room_type');
        return false;
    }
    if (!guests || guests < 1 || guests > 10) {
        highlightField('num_guests');
        return false;
    }

    return true;
}

/**
 * Highlight a field that failed validation.
 */
function highlightField(id) {
    const el = document.getElementById(id);
    if (!el) return;

    el.style.borderColor = '#ef5350';
    el.style.boxShadow = '0 0 0 3px rgba(239, 83, 80, 0.2)';

    setTimeout(() => {
        el.style.borderColor = '';
        el.style.boxShadow = '';
    }, 2000);

    el.focus();
}

/**
 * Render prediction results.
 */
function renderResults(data) {
    renderPriceHero(data);
    renderModelCards(data.predictions);
    renderHotelCards(data.recommendations);
}

/**
 * Render the main price display.
 */
function renderPriceHero(data) {
    const priceValue = document.getElementById('avgPrice');
    const priceMeta = document.getElementById('priceMeta');

    if (priceValue) {
        // Animate the price counter
        animateCounter(priceValue, data.average_price);
    }

    if (priceMeta) {
        priceMeta.innerHTML = `
            <span class="tag">📍 ${data.input.city}, ${data.input.state}</span>
            <span class="tag">🏨 ${data.input.room_type}</span>
            <span class="tag">👥 ${data.input.num_guests} Guest${data.input.num_guests > 1 ? 's' : ''}</span>
            <span class="tag">🌤️ ${data.input.season}</span>
            ${data.input.is_weekend ? '<span class="tag">📅 Weekend</span>' : ''}
            ${data.input.has_events ? '<span class="tag">🎉 Local Events</span>' : ''}
        `;
    }
}

/**
 * Animate a counter from 0 to target value.
 */
function animateCounter(element, target) {
    const duration = 1200;
    const start = performance.now();
    const initial = 0;

    function update(currentTime) {
        const elapsed = currentTime - start;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = initial + (target - initial) * eased;

        element.textContent = `₹${Math.round(current).toLocaleString('en-IN')}`;

        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            element.textContent = `₹${target.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
        }
    }

    requestAnimationFrame(update);
}

/**
 * Render model comparison cards.
 */
function renderModelCards(predictions) {
    const container = document.getElementById('modelCards');
    if (!container) return;

    const icons = ['📈', '🌲', '🚀'];
    const modelNames = Object.keys(predictions);
    const prices = Object.values(predictions);
    const maxPrice = Math.max(...prices);

    container.innerHTML = modelNames.map((name, i) => {
        const price = predictions[name];
        const percentage = (price / maxPrice) * 100;

        return `
            <div class="model-card glass-card">
                <div class="model-icon">${icons[i]}</div>
                <div class="model-name">${name}</div>
                <div class="model-price">₹${price.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</div>
                <div class="model-bar">
                    <div class="model-bar-fill" style="width: 0%;" data-width="${percentage}%"></div>
                </div>
            </div>
        `;
    }).join('');

    // Animate bars
    setTimeout(() => {
        container.querySelectorAll('.model-bar-fill').forEach(bar => {
            bar.style.width = bar.dataset.width;
        });
    }, 100);
}

/**
 * Render hotel recommendation cards.
 */
function renderHotelCards(hotels) {
    const container = document.getElementById('hotelCards');
    if (!container) return;

    container.innerHTML = hotels.map((hotel, i) => {
        const stars = '★'.repeat(Math.round(hotel.rating)) + '☆'.repeat(5 - Math.round(hotel.rating));

        return `
            <div class="hotel-card glass-card">
                <div class="hotel-rank">#${i + 1}</div>
                <div class="hotel-name">${hotel.name}</div>
                <div class="hotel-location">📍 ${hotel.city}, ${hotel.state}</div>
                <div class="hotel-details">
                    <div>
                        <div class="hotel-price-label">Price / Night</div>
                        <div class="hotel-price">₹${hotel.price.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</div>
                    </div>
                    <div class="hotel-rating">
                        <span class="stars">${stars}</span>
                        <span class="rating-text">${hotel.rating}</span>
                        <span class="reviews">(${hotel.reviews.toLocaleString()})</span>
                    </div>
                </div>
                <span class="hotel-room-type">${hotel.room_type}</span>
            </div>
        `;
    }).join('');
}

/**
 * Show an error message.
 */
function showError(message) {
    // Create a toast notification
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        top: 24px;
        right: 24px;
        background: rgba(239, 83, 80, 0.15);
        border: 1px solid rgba(239, 83, 80, 0.3);
        backdrop-filter: blur(20px);
        border-radius: 14px;
        padding: 16px 24px;
        color: #ef9a9a;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        z-index: 1000;
        animation: fadeUp 0.3s ease-out;
        max-width: 400px;
    `;
    toast.textContent = `⚠️ ${message}`;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(-10px)';
        toast.style.transition = 'all 0.3s ease-in';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
