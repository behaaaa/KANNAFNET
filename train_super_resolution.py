def train_super_resolution_model(
    hr_data_dir,
    num_epochs=50,
    batch_size=16,
    patch_size=128,
    lr_scale=4,
    learning_rate=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    model_save_path="trained_sr_model.pth"
):
    # 1. Create data loaders from high-resolution image directory
    train_loader, val_loader = create_dataloaders(
        hr_dir=hr_data_dir,
        batch_size=batch_size,
        patch_size=patch_size,
        lr_scale=lr_scale,
        num_workers=4
    )

    # 2. Initialize model and loss functions
    model = KAN_NAF_SRModel(upscale=lr_scale).to(device)
    criterion_charb = CharbonnierLoss()                   # Robust L1 loss
    criterion_percep = PerceptualLoss().to(device)        # VGG-based perceptual loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')  # For saving best model

    # 3. Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)

            sr = model(lr)  # Super-resolved image

            # Combine Charbonnier and perceptual losses
            loss_charb = criterion_charb(sr, hr)
            loss_percep = criterion_percep(sr, hr)
            loss = loss_charb + 0.1 * loss_percep

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # 4. Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                lr = batch['lr'].to(device)
                hr = batch['hr'].to(device)
                sr = model(lr)

                loss_charb = criterion_charb(sr, hr)
                loss_percep = criterion_percep(sr, hr)
                loss = loss_charb + 0.1 * loss_percep

                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        logger.info(
            f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # 5. Save the model with the best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"✅ Best model saved at epoch {epoch+1} with val loss {avg_val_loss:.4f}")

    logger.info("Training complete!")
    return model
