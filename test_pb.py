import progressbar
bar = progressbar.ProgressBar(max_value=14).start()


for idx in range(14):
	print(idx)
	bar.update(idx+1)

bar.finish()
