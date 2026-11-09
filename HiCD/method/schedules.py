def delayed_linear_warmup_scale(
    epoch: int,
    *,
    start_epoch: int,
    warmup_epochs: int,
) -> float:


    epoch = int(epoch)
    start_epoch = int(start_epoch)
    warmup_epochs = int(warmup_epochs)
    if epoch < 1:
        raise ValueError("epoch must be at least 1")
    if start_epoch < 1:
        raise ValueError("start_epoch must be at least 1")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    if epoch < start_epoch:
        return 0.0
    if warmup_epochs == 0:
        return 1.0
    return min(
        float(epoch - start_epoch + 1) / float(warmup_epochs),
        1.0,
    )
