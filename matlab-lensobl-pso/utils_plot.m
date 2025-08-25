function utils_plot(values, plot_title)
if isempty(values)
	return;
end
figure('Name', plot_title, 'NumberTitle', 'off');
semilogy(values, 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Best fitness (log)');
grid on;
title(plot_title);
end
