import os
import argparse
from model import build_transfer_unet
from utils import data_generator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

def train(data_dir, epochs=50, batch_size=16, model_path='colorization_model.keras'):
    """
    Trains the Transfer Learning model on images in data_dir.
    """
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist. Please provide a path to images.")
        return

    num_images = len([f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {num_images} images for training.")
    
    if num_images == 0:
        print("No images found in the directory.")
        return

    # 1. Build Model
    model = build_transfer_unet()
    
    # 2. Callbacks
    checkpoint = ModelCheckpoint(model_path, monitor='loss', save_best_only=True, verbose=1)
    lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1)

    # 3. Phase 1: Training Decoder Only (Encoder frozen)
    print("\n--- Phase 1: Training Decoder (Encoder Frozen) ---")
    train_gen = data_generator(data_dir, batch_size=batch_size)
    steps_per_epoch = max(1, num_images // batch_size)

    try:
        model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=min(10, epochs),
            callbacks=[checkpoint, lr_reducer]
        )
        
        # 4. Phase 2: Fine-tuning (Unfreeze a bit if requested)
        if epochs > 10:
            print("\n--- Phase 2: Fine-tuning (Unfreezing Encoder) ---")
            # Unfreeze the base model
            model.trainable = True
            # Re-compile with a lower learning rate
            from tensorflow.keras.optimizers import Adam
            model.compile(optimizer=Adam(1e-5), loss='mse', metrics=['accuracy'])
            
            model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs - 10,
                callbacks=[checkpoint, lr_reducer]
            )
            
        print(f"Training complete. Model saved to {model_path}")
    except KeyboardInterrupt:
        print("Training interrupted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transfer Learning Colorization Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to folder containing colored images")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--model_path", type=str, default="colorization_model.keras", help="Path to save the model")

    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.batch_size, args.model_path)
