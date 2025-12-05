// Emotion emoji mapping
const emotionEmojis = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'surprise': '😲',
    'fear': '😨',
    'disgust': '🤢',
    'neutral': '😐'
};

// Emotion French translations
const emotionTranslations = {
    'happy': 'Joyeux',
    'sad': 'Triste',
    'angry': 'En colère',
    'surprise': 'Surpris',
    'fear': 'Peur',
    'disgust': 'Dégoût',
    'neutral': 'Neutre'
};

// Update emotion display
function updateEmotion() {
    fetch('/emotion')
        .then(response => response.json())
        .then(data => {
            const emotion = data.emotion.toLowerCase();
            const confidence = data.confidence;
            const probabilities = data.probabilities || {};

            // Update emoji
            const emotionIcon = document.getElementById('emotionIcon');
            emotionIcon.textContent = emotionEmojis[emotion] || '😊';

            // Update label
            const emotionLabel = document.getElementById('emotionLabel');
            emotionLabel.textContent = emotionTranslations[emotion] || emotion;

            // Update confidence bar
            const confidenceFill = document.getElementById('confidenceFill');
            confidenceFill.style.width = `${confidence}%`;

            // Update confidence text
            const confidenceText = document.getElementById('confidenceText');
            confidenceText.textContent = `${confidence.toFixed(1)}%`;

            // Add animation class
            emotionIcon.style.animation = 'none';
            setTimeout(() => {
                emotionIcon.style.animation = 'pulse 2s ease-in-out infinite';
            }, 10);

            // Update comparison bars
            updateComparisonBars(probabilities, emotion);
        })
        .catch(error => {
            console.error('Error fetching emotion:', error);
        });
}

// Update comparison bars
function updateComparisonBars(probabilities, currentEmotion) {
    const container = document.getElementById('comparisonBars');
    if (!container) return;

    // Sort emotions by probability (descending)
    const sortedEmotions = Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1]);

    // Clear existing bars
    container.innerHTML = '';

    // Create bars for each emotion
    sortedEmotions.forEach(([emotion, probability]) => {
        const barContainer = document.createElement('div');
        barContainer.className = 'comparison-bar-container';

        const isActive = emotion === currentEmotion;
        if (isActive) {
            barContainer.classList.add('active');
        }

        const label = document.createElement('div');
        label.className = 'comparison-label';
        label.innerHTML = `${emotionEmojis[emotion]} ${emotionTranslations[emotion]}`;

        const barWrapper = document.createElement('div');
        barWrapper.className = 'comparison-bar-wrapper';

        const bar = document.createElement('div');
        bar.className = 'comparison-bar';
        bar.style.width = `${probability}%`;

        const value = document.createElement('span');
        value.className = 'comparison-value';
        value.textContent = `${probability.toFixed(1)}%`;

        barWrapper.appendChild(bar);
        barWrapper.appendChild(value);

        barContainer.appendChild(label);
        barContainer.appendChild(barWrapper);
        container.appendChild(barContainer);
    });
}

// Check application health
function checkHealth() {
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            if (!data.camera || !data.model) {
                console.warn('Application health check failed:', data);
            }
        })
        .catch(error => {
            console.error('Error checking health:', error);
        });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('🎀 Application de Reconnaissance d\'Émotions initialisée 🎀');

    // Check health on load
    checkHealth();

    // Update emotion every 500ms
    setInterval(updateEmotion, 500);

    // Add smooth scroll behavior
    document.documentElement.style.scrollBehavior = 'smooth';

    // Add loading animation to video
    const videoStream = document.getElementById('videoStream');
    videoStream.addEventListener('load', () => {
        console.log('✨ Flux vidéo chargé avec succès');
    });

    videoStream.addEventListener('error', () => {
        console.error('❌ Erreur lors du chargement du flux vidéo');
    });
});

// Add sparkle effect on click
document.addEventListener('click', (e) => {
    const sparkle = document.createElement('div');
    sparkle.style.position = 'fixed';
    sparkle.style.left = e.clientX + 'px';
    sparkle.style.top = e.clientY + 'px';
    sparkle.style.pointerEvents = 'none';
    sparkle.style.fontSize = '20px';
    sparkle.textContent = '✨';
    sparkle.style.animation = 'sparkleFloat 1s ease-out forwards';
    sparkle.style.zIndex = '9999';

    document.body.appendChild(sparkle);

    setTimeout(() => {
        sparkle.remove();
    }, 1000);
});

// Add sparkle animation
const style = document.createElement('style');
style.textContent = `
    @keyframes sparkleFloat {
        0% {
            opacity: 1;
            transform: translate(0, 0) scale(1);
        }
        100% {
            opacity: 0;
            transform: translate(0, -50px) scale(0.5);
        }
    }
`;
document.head.appendChild(style);
