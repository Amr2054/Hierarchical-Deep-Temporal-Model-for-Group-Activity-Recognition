import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter


def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, run_dir, save_name,
                       logger, log_interval=10, class_names=None):
    """
    Generic training and validation loop.
    Saves the best model weights, metrics and logs to the specified run directory.
    """
    best_val_acc = 0.0

    # Initialize the Gradient Scaler for AMP
    scaler = GradScaler('cuda')

    tb_log_dir = os.path.join(run_dir, 'tensorboard_logs')
    writer = SummaryWriter(log_dir=tb_log_dir)
    save_path = os.path.join(run_dir, save_name)

    logger.info(f"All outputs will be saved to: {run_dir}")

    for epoch in range(num_epochs):
        logger.info(f"\n{'=' * 15} Epoch {epoch + 1}/{num_epochs} {'=' * 15}")

        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0


        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Use the scaler for backward pass and optimization
            scaler.scale(loss).backward() # Multiplies the loss by the scale factor to prevent underflow of gradients
            scaler.step(optimizer) # unscale gradients back to normal scale to update weights
            scaler.update()


            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)

            batch_correct = (predicted == labels).sum().item()
            train_correct += batch_correct


            if (i + 1) % log_interval == 0 or (i + 1) == len(train_loader):
                current_acc = 100.0 * batch_correct / batch_size
                logger.info(
                    f"Epoch: {epoch + 1} | Batch: {i + 1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%")

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        all_preds = []
        all_labels = []

        logger.info("Evaluating validation set")

        with torch.no_grad():

            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total

        # EPOCH LOGGING & METRICS
        logger.info(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        logger.info(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.2f}%")

        # Write to TensorBoard
        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_train_acc, epoch)
        writer.add_scalar('Loss/Validation', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', epoch_val_acc, epoch)

        logger.info("\n Validation Metrics Breakdown")
        # classification_report returns a string, so just pass it to the logger
        logger.info("\n" + classification_report(all_labels, all_preds, zero_division=0))

        # Save the best model and plot the Confusion Matrix
        if epoch_val_acc > best_val_acc:
            logger.info(
                f"** Validation accuracy improved from {best_val_acc:.2f}% to {epoch_val_acc:.2f}%. Saving model... **")
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), save_path)

            # Generate Confusion Matrix
            if class_names is None:
                class_names = [str(c) for c in range(len(set(all_labels)))]

            cm = confusion_matrix(all_labels, all_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

            # Plot and save
            fig, ax = plt.subplots(figsize=(10, 10))
            disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
            plt.title(f"Best Validation Confusion Matrix (Epoch {epoch + 1})")
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, 'best_confusion_matrix.png'), dpi=300)
            plt.close(fig)

    # Close TensorBoard writer
    writer.close()
    return model


def print_model_summary(model):
    """
    Prints a clean summary of a PyTorch model's trainable and non-trainable parameters.
    """
    print("\n" + "=" * 40)
    print(f"{'MODEL SUMMARY':^40}")
    print("=" * 40)

    total_params = 0
    trainable_params = 0

    for name, parameter in model.named_parameters():
        params_count = parameter.numel()
        total_params += params_count
        if parameter.requires_grad:
            trainable_params += params_count

    non_trainable_params = total_params - trainable_params

    # Format numbers with commas for readability
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"Frozen Parameters:     {non_trainable_params:,}")

    # Calculate percentage
    if total_params > 0:
        percent_trainable = (trainable_params / total_params) * 100
        print(f"% Trainable:           {percent_trainable:.2f}%")
    print("=" * 40 + "\n")