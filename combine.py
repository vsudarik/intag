with open('smd1_2_qiP6R.txt', 'r') as infile, open('smd1_qiP6R.txt', 'a') as outfile:
    for line in infile:
        columns = line.split()

        # Проверка, что первый столбец содержит число
        try:
            first_column = float(columns[0])
            fourth_column = float(columns[3])
        except ValueError:
            print(f"Невозможно обработать строку: {line.strip()}")
            continue

        # Прибавляем константу к первому столбцу
        first_column += 75054000
        fourth_column += 6.3799205573478525721 

        # Заменяем первый столбец и объединяем столбцы обратно
        columns[0] = str(first_column)
        columns[3] = str(fourth_column)
        new_line = ' '.join(columns)

        # Записываем новую строку в выходной файл
        outfile.write(new_line + '\n')
