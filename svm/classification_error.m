function err = classification_error(y_pred, y_true)
	err = 1 - length(find(y_pred == y_true)) / length(y_true);
end
