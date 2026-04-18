def bankers_algorithm(available, maximum, allocation, process_id=None, request=None):
	avail = available[:]
	alloc = [row[:] for row in allocation]

	n_processes = len(maximum)
	n_resources = len(avail)

	need = [
		[maximum[i][j] - alloc[i][j] for j in range(n_resources)]
		for i in range(n_processes)
	]

	if process_id is not None and request is not None:
		for j in range(n_resources):
			if request[j] > need[process_id][j]:
				return {
					"safe": False,
					"sequence": [],
					"granted": False,
					"message": "Error: request exceeds process need.",
				}
			if request[j] > avail[j]:
				return {
					"safe": False,
					"sequence": [],
					"granted": False,
					"message": "Wait: resources not currently available.",
				}

		for j in range(n_resources):
			avail[j] -= request[j]
			alloc[process_id][j] += request[j]
			need[process_id][j] -= request[j]

	work = avail[:]
	finish = [False] * n_processes
	sequence = []

	while len(sequence) < n_processes:
		progressed = False
		for i in range(n_processes):
			if finish[i]:
				continue
			if all(need[i][j] <= work[j] for j in range(n_resources)):
				for j in range(n_resources):
					work[j] += alloc[i][j]
				finish[i] = True
				sequence.append(i)
				progressed = True
		if not progressed:
			if process_id is not None and request is not None:
				return {
					"safe": False,
					"sequence": [],
					"granted": False,
					"message": "Request denied: system would become unsafe.",
				}
			return {
				"safe": False,
				"sequence": [],
				"granted": None,
				"message": "System is unsafe.",
			}

	msg = "System is safe."
	granted = None
	if process_id is not None and request is not None:
		msg = "Request granted. System remains safe."
		granted = True

	return {
		"safe": True,
		"sequence": sequence,
		"granted": granted,
		"message": msg,
	}
