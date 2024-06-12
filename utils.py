def calc_accuracy(outputs) -> float:
    count = 0
    total = 0
    for target, pred in zip(outputs["targets"], outputs["preds"]):
        target = target[0]
        pred = pred[0]

        total += 1
        if input == pred:
            count += 1
    return count / total
