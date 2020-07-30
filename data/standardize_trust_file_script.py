fr = open('epinions_trust_data.txt', 'r')
fw = open('standardized_epinions_trust_data.txt', 'w')
for line in fr:
	fw.write(line.lstrip())
fr.close()
fw.close()
