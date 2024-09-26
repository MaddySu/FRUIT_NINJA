import cv2
import mediapipe as mp
import random
import numpy as np
import pygame
import time

# Constants
FRUIT_SIZE = (170, 170)  # Width, Height
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080  # Updated window size
FRUIT_SPAWN_PROBABILITY = 0.05  # Probability of spawning a fruit per frame
GAME_DURATION = 3 * 60  # 15 minutes in seconds

# Load and resize images
def load_and_resize_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

# Load images
fruits = ["apple", "banana", "coconut", "orange", "pineapple"]
fruit_images = {}
cut_fruit_images = {}

for fruit in fruits:
    fruit_images[fruit] = load_and_resize_image(f'{fruit}.png', FRUIT_SIZE)
    cut_fruit_images[fruit] = load_and_resize_image(f'cut_{fruit}.png', FRUIT_SIZE)

# Load and resize the image to overlay on the window
window_overlay_image = load_and_resize_image('bg1.png', (1920, 1080))
finger_image = load_and_resize_image('2.png', (140, 140))  # Example size

# Initialize pygame for sound
pygame.mixer.init()
cut_sound = pygame.mixer.Sound('sound1.wav')

# Hand tracking setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.2)
mp_drawing = mp.solutions.drawing_utils

# Function to overlay image with alpha channel
def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) with `alpha_mask`."""
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, 3] / 255.0
    alpha_inv = 1.0 - alpha

    for c in range(0, 3):
        img_crop[..., c] = (alpha * img_overlay_crop[..., c] +
                            alpha_inv * img_crop[..., c])

# Function to rotate an image
def rotate_image(image, angle):
    """Rotate the image by the given angle."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    return rotated

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

# Initialize score and game state
score = 0
game_start = False
start_time = None
angle = 0  # Initialize rotation angle

# Main game loop
fruits_on_screen = []

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Detect hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finger_position = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_position = (
                int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * SCREEN_WIDTH),
                int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * SCREEN_HEIGHT)
            )

    # Overlay image at (0, 0)
    overlay_image_alpha(frame, window_overlay_image, 0, 0, window_overlay_image)

    if finger_position:
        # Rotate the finger image
        rotated_finger_image = rotate_image(finger_image, angle)
        angle += 20  # Increment the angle for the next frame
        if angle >= 360:
            angle = 0

        # Overlay the rotated finger image on top of the orange image
        overlay_image_alpha(frame, rotated_finger_image, finger_position[0] - rotated_finger_image.shape[1] // 2, finger_position[1] - rotated_finger_image.shape[0] // 2, rotated_finger_image)

    # Game start/stop logic
    key = cv2.waitKey(1)
    if key == ord('s') and not game_start:
        game_start = True
        start_time = time.time()

        score = 0  # Reset the score when starting the game
    elif key == 27:  # ESC key to quit
        break

    # Game over logic
    if game_start and time.time() - start_time > GAME_DURATION:
        game_start = False
        # Load and resize the "over.png" image
        game_over_image = load_and_resize_image('5.png', (SCREEN_WIDTH, SCREEN_HEIGHT))
        # Overlay the "over.png" image on the frame
        overlay_image_alpha(frame, game_over_image, 0, 0, game_over_image)
        # Display final score on top of the image
        cv2.putText(frame, f"{score}", (600, 430), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
        cv2.imshow('Fruit Ninja', frame)
        cv2.waitKey(5000)  # Show final score for 5 seconds

    if game_start:
        # Calculate remaining time
        elapsed_time = time.time() - start_time
        remaining_time = int(GAME_DURATION - elapsed_time)

        # Convert remaining time to minutes and seconds
        minutes = remaining_time // 60
        seconds = remaining_time % 60
        timer_text = f"Time: {minutes:02}:{seconds:02}"

        # Display timer

        cv2.putText(frame, timer_text, (1500, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 8)
        cv2.putText(frame, timer_text, (1500, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)

        # Display and move fruits
        for fruit in fruits_on_screen:
            if fruit['cut']:
                overlay_image_alpha(frame, cut_fruit_images[fruit['type']], fruit['x'], fruit['y'], cut_fruit_images[fruit['type']])
            else:
                overlay_image_alpha(frame, fruit_images[fruit['type']], fruit['x'], fruit['y'], fruit_images[fruit['type']])

            fruit['y'] += fruit['speed']

            # Check if finger cuts the fruit
            if finger_position and not fruit['cut']:
                if (fruit['x'] < finger_position[0] < fruit['x'] + FRUIT_SIZE[0] and
                        fruit['y'] < finger_position[1] < fruit['y'] + FRUIT_SIZE[1]):
                    fruit['cut'] = True
                    cut_sound.play()  # Play sound when fruit is cut
                    score += 1  # Increment score when fruit is cut

        # Remove fruits that have fallen off the screen
        fruits_on_screen = [fruit for fruit in fruits_on_screen if fruit['y'] < SCREEN_HEIGHT]

        # Add new fruits at random intervals
        if random.random() < FRUIT_SPAWN_PROBABILITY:
            num_fruits = random.randint(1, 3)  # Spawn between 1 and 3 fruits at a time
            for _ in range(num_fruits):
                fruits_on_screen.append({
                    'type': random.choice(fruits),
                    'x': random.randint(0, SCREEN_WIDTH - FRUIT_SIZE[0]),
                    'y': 0,
                    'speed': random.randint(10, 35),  # Random speed between 5 and 15
                    'cut': False
                })

    # Overlay score text onto the frame
    cv2.putText(frame, f"Score: {score}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 8)
    cv2.putText(frame, f"Score: {score}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (250, 250, 250), 4)

    cv2.imshow('Fruit Ninja', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
