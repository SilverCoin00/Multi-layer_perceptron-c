#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Supporting build:

#define Nan (0.0f/0.0f)
static struct Node {
	int data;
	struct Node* next;
};
typedef struct Node Node;
typedef struct Data_Frame {
    char** features;
    float** data;
	char*** str_data;
	int* str_cols;
    int row, col;          // row, col of float data
} Data_Frame;
static struct Space {
    int key;
	char* keys;
	struct Space* next;
};
typedef struct Space Space;
typedef struct Set {
    Space** arr;
	unsigned max_size;
	int size;
} Set;

static Node* new_node(int val) {
	Node* newn = (Node*)malloc(sizeof(Node));
	newn->data = val;
	newn->next = NULL;
	return newn;
}
static void add_node(Node** head, int val) {
	if (*head == NULL) {
		*head = new_node(val);
		return ;
	}
	Node* cur = *head;
	while (cur->next != NULL) cur = cur->next;
	cur->next = new_node(val);
}
static void free_llist(Node** head) {
	Node* cur = *head;
	while (*head != NULL) {
		*head = (*head)->next;
		free(cur);
		cur = *head;
	}
}
static void swap(float* a, float* b) {
	float t = *a;
	*a = *b;
	*b = t;
}
static void heapify_tree(float* s, int size, int i, int is_max_heap) {  // heapify down
	int cur = i, left = 2*i + 1, right = 2*i + 2;
	if (size == 1 || size == 0) {
		return ;
	} else if (is_max_heap == 1) {
		if (left < size && s[left] > s[cur]) cur = left;
		if (right < size && s[right] > s[cur]) cur = right;
	} else if (is_max_heap == 0) {
		if (left < size && s[left] < s[cur]) cur = left;
		if (right < size && s[right] < s[cur]) cur = right;
	}
	if (cur != i) {
		swap(&s[i], &s[cur]);
		heapify_tree(s, size, cur, is_max_heap);
	}
}
static void heap_add(float* s, int* size, int val, int is_max_heap) {
    int i = (*size)++, parent;
    s[i] = val;
    while (i > 0) {
        parent = (i - 1) / 2;
        if ((is_max_heap && s[i] > s[parent]) || (!is_max_heap && s[i] < s[parent])) {  // heapify up
            swap(&s[i], &s[parent]);
            i = parent;
        } else break;
    }
}
static void heap_remove(float* s, int* size, int val, int is_max_heap) {
    int i;
    for (i = 0; i < *size && val != s[i]; i++) ;
	if (i == *size) return ;
    swap(&s[i], &s[--(*size)]);
    heapify_tree(s, *size, i, is_max_heap);
}
static int is_prime(int n) {
	if(n < 2) return 0;
    for(int i = 2; i*i <= n; i++){
        if(n % i == 0) {
            return 0;
        }
    }
    return 1;
}
static void trans_prime(int* n) {
	if (*n % 2 == 0) (*n)++;
	while (!is_prime(*n)) *n += 2;
}
static int hash_func(const char* keys, int max_size) {
    if (!keys) return 0;
	int hash = 0;
	while (*keys) hash = (hash* 31) + (*keys++);
	return hash % max_size;
}
Set* new_set(int max_size) {
	Set* st = (Set*)malloc(sizeof(Set));
	trans_prime(&max_size);
	st->max_size = max_size;
	st->size = 0;
	st->arr = (Space**)malloc(st->max_size* sizeof(Space*));
	for (int i = 0; i < st->max_size; i++) {
		st->arr[i] = NULL;
	}
	return st;
}
void set_add(Set* st, const char* keys, int key) {
	int id = hash_func(keys, st->max_size);
	Space* head = st->arr[id];
	Space* mid = head;
	while (mid != NULL) {
		if (!strcmp(mid->keys, keys)) return ;
		mid = mid->next;
	}
	Space* news = (Space*)malloc(sizeof(Space));
	news->keys = strdup(keys);
    news->key = key;
	news->next = head;
	st->arr[id] = news;
	st->size++;
}
int set_find(Set* st, const char* keys) {
    int id = hash_func(keys, st->max_size);
    Space* mid = st->arr[id];
    while (mid != NULL) {
        if (!strcmp(mid->keys, keys)) return 1;
        mid = mid->next;
	}
    return 0;
}
int set_call(Set* st, const char* keys) {
    int id = hash_func(keys, st->max_size);
    Space* mid = st->arr[id];
    while (mid != NULL) {
        if (!strcmp(mid->keys, keys)) return mid->key;
        mid = mid->next;
	}
    return -1;
}
int* set_key_access(Set* st, const char* keys) {
    int id = hash_func(keys, st->max_size);
    Space* mid = st->arr[id];
    while (mid != NULL) {
        if (!strcmp(mid->keys, keys)) return &(mid->key);
        mid = mid->next;
	}
    return NULL;
}
void free_set(Set* st) {
	if (!st) return ;
    Space* mid, * next;
	for (int i = 0; i < st->max_size; i++) {
        mid = st->arr[i];
        while (mid != NULL) {
            next = mid->next;
            if (mid->keys) free(mid->keys);
            free(mid);
            mid = next;
        }
    }
    free(st->arr);
    free(st);
}

// Numpy:

float** new_matrix(int row, int col) {
	float** newm = (float**)calloc(row, sizeof(float*));
	for (int i = 0; i < row; i++) newm[i] = (float*)calloc(col, sizeof(float));
	return newm;
}
float** matrix_multiply(float** a, float** b, int row_a, int col_a, int col_b) {
	float** res = (float**)malloc(row_a* sizeof(float*));
	for (int i = 0, j, k; i < row_a; i++) {
		res[i] = (float*)calloc(col_b, sizeof(float));
		for (j = 0; j < col_b; j++) {
			for (k = 0; k < col_a; k++) res[i][j] += a[i][k]* b[k][j];
		}
	}
	return res;
}
float** transpose_matrix(float** matrix, int row, int col) {
	float** trp = (float**)malloc(col* sizeof(float*));
	for (int i = 0, j; i < col; i++) {
		trp[i] = (float*)malloc(row* sizeof(float));
		for (j = 0; j < row; j++) trp[i][j] = matrix[j][i];
	}
	return trp;
}
float** element_wise_multiply(float** a, float** b, int row, int col) {
	float** newm = (float**)malloc(row* sizeof(float*));
	for (int i = 0, j; i < row; i++) {
		newm[i] = (float*)malloc(col* sizeof(float));
		for (j = 0; j < col; j++) newm[i][j] = a[i][j]* b[i][j];
	}
	return newm;
}
float** copy_matrix(float** matrix, int row, int col) {
	float** copy = (float**)malloc(row* sizeof(float*));
	for (int i = 0, j; i < row; i++) {
		copy[i] = (float*)malloc(col* sizeof(float));
		for (j = 0; j < col; j++) copy[i][j] = matrix[i][j];
	}
	return copy;
}
void print_matrix(float** matrix, int row, int col, int decimal) {
    int i, j;
	for (i = 0; i < row; i++) {
		if (i != 0) printf("\n [");
        else printf("[[");
		printf("%.*f", decimal, matrix[i][0]);
		for (j = 1; j < col; j++) printf(", %.*f", decimal, matrix[i][j]);
		printf("]");
	}
	printf("]\n");
}
void free_matrix(float** matrix, int row) {
	if (!matrix) return ;
	for (int i = 0; i < row; i++) free(matrix[i]);
	free(matrix);
}
float mean(float* y, int length) {
	float total = 0.0;
	for (int i = 0; i < length; i++) total += y[i];
	return total / length;
}
float median(float* s, int length) {
	if (length == 1) return s[0];
	float median;
	int as = 0, is = 0;
	float* max_heap = (float*)calloc((length / 2 + 2), sizeof(float));
	float* min_heap = (float*)calloc((length / 2 + 2), sizeof(float));
	max_heap[as++] = s[0];
	for (int i = 1; i < length; i++) {
		if (as == is) {
			if (s[i] > min_heap[0]) {
				heap_add(max_heap, &as, min_heap[0], 1);
				heap_remove(min_heap, &is, min_heap[0], 0);
				heap_add(min_heap, &is, s[i], 0);
			} else heap_add(max_heap, &as, s[i], 1);
		} else {  // as > is
			if (s[i] < max_heap[0]) {
				heap_add(min_heap, &is, max_heap[0], 0);
				heap_remove(max_heap, &as, max_heap[0], 1);
				heap_add(max_heap, &as, s[i], 1);
			} else heap_add(min_heap, &is, s[i], 0);
		}
	}
	if (as == is) median = (max_heap[0] + min_heap[0]) / 2;
	else median = max_heap[0];
	free(max_heap);
	free(min_heap);
	return median;
}
float sum_square_error(float* y_pred, float* y_true, int length) {
	float sum = 0.0;
	for (int i = 0; i < length; i++) sum += (y_pred[i] - y_true[i])*(y_pred[i] - y_true[i]);
	return sum;
}
float mean_square_error(float* y_pred, float* y_true, int length) {
	return sum_square_error(y_pred, y_true, length) / length;
}
float sqroot(float num, float error) {
	if (num < 0) return 0;
	float sqr = num / 2;
	while (sqr*sqr - num > error || num - sqr*sqr > error) sqr = (sqr + num / sqr) / 2;
	return sqr;
}

// Pandas:

static int is_blank(char* s) {
	for (int i = 0; s[i]; i++) if (s[i] != ' ' && s[i] != '\t' && s[i] != '\n' && s[i] != '\r') return 0;;
	return 1;
}
static int is_nan(char* s) {
	if (!s) return 0;
	s[strcspn(s, "\n")] = '\0';
	const char* nan[] = {"NA", "NAN", "NaN", "N/A", "nan", "null", "NULL", "None", "missing", "?", "--", "-", ""};
	for (int i = 0; i < 13; i++) {
		if (!strcmp(s, nan[i])) return 1;
	}
	return 0;
}
static int count_row(FILE* file, int max_line_length) {
	char* s = (char*)malloc(max_line_length* sizeof(char));
	int count;
	for (count = 0; fgets(s, max_line_length, file); ) if (!is_blank(s)) count++;;
	free(s);
	rewind(file);
	return count;
}
static int count_col(FILE* file, int max_line_length, char* seperate) {
	char* s = (char*)malloc(max_line_length* sizeof(char));
	if (fgets(s, max_line_length, file) == NULL) {
		free(s);
		return 0;
	}
	int count = 1;
	for (int i = 0; s[i] != '\n'; i++) {
		if (s[i] == seperate[0]) count++;
	}
	free(s);
	rewind(file);
	return count;
}
static void find_string_col(FILE* file, int max_line_length, char* seperate, int** str_cols) {
	char* s = (char*)malloc(max_line_length* sizeof(char));
	if (fgets(s, max_line_length, file) == NULL) {
		free(s);
		*str_cols = (int*)malloc(sizeof(int));
		(*str_cols)[0] = 0;
		return ;
	}
	fgets(s, max_line_length, file);    // constrained first data line is the standard
	Node* list_char = NULL;
	int count = 0;
	for (int i = 0, chk = 0, col = 0; s[i]; i++) {
		if (((s[i] >= 65 && s[i] <= 90) || (s[i] >= 97 && s[i] <= 122)) && chk == 0 && s[i] != seperate[0]) {
			count++;
			chk = 1;
			add_node(&list_char, col);
		}
		if (s[i] == seperate[0]) {
			chk = 0;
			col++;
		}
	}
	*str_cols = (int*)malloc((count + 1)* sizeof(int));
	(*str_cols)[0] = count;
	Node* mid = list_char;
	for (int i = 0; i < count; i++, mid = mid->next) (*str_cols)[i + 1] = mid->data;
	free(s);
	free_llist(&list_char);
	rewind(file);
}
static int is_data_line(char* s, char* seperate) {
	for (int i = 0; s[i] != seperate[0]; i++) {
		if ((s[i] < 48 || s[i] > 57) && s[i] != '.' && s[i] != '-') return 0;
	}
	return 1;
}
int strtoi(const char* number) {
	int num = 0;
	for (int i = 0; number[i]; i++) {
		if (number[i] < 48 || number[i] > 57) return -1;
		num *= 10;
		num += number[i] - 48;
	}
	return num;
}
void ftostr(float num, char* s, int num_nums) {
	int intg, i, j;
	for (j = 0; num >= 1.0; num /= 10, j++);
	for (i = 0, num *= 10 ; i < num_nums; i++) {
		if (i == j) {
			s[i] = '.';
			num_nums++;
			continue;
		}
		intg = (int) num;
		s[i] = intg % 10 + '0';
		num *= 10;
	}
	s[i] = '\0';
}
Data_Frame* read_csv(char* file_name, int max_line_length, char* seperate) {
    FILE* file = fopen(file_name, "r");
	if (!file) {
		printf("Error: Cannot open file !!");
		return NULL;
	}
	int i, j, k, size;
    Data_Frame* newd = (Data_Frame*)malloc(sizeof(Data_Frame));
	newd->row = count_row(file, max_line_length) - 1;
	find_string_col(file, max_line_length, seperate, &(newd->str_cols));
    newd->col = count_col(file, max_line_length, seperate) - newd->str_cols[0];
	newd->features = (char**)malloc(newd->col* sizeof(char*));
	newd->data = (float**)malloc(newd->row* sizeof(float*));
	if (newd->str_cols[0] != 0) newd->str_data = (char***)malloc(newd->row* sizeof(char**));

    char* s = (char*)malloc(max_line_length* sizeof(char));
	char* token;
    fgets(s, max_line_length, file);
    if (!is_data_line(s, seperate)) {
		s[strcspn(s, "\n")] = '\0';
		token = strtok(s, seperate);
		j = newd->col + newd->str_cols[0];
		for (i = 0; token != NULL && i < j; i++) {
			size = strlen(token) + 1;
			newd->features[i] = (char*)malloc(size* sizeof(char));
			snprintf(newd->features[i], size, "%s", token);
			token = strtok(NULL, seperate);
		}
	} else rewind(file);

	float num;
	for (i = 0, j = 0; i < newd->row; i++, j = 0) {
		newd->data[i] = (float*)malloc(newd->col* sizeof(float));
		if (newd->str_cols[0] != 0) newd->str_data[i] = (char**)malloc(newd->str_cols[0]* sizeof(char*));
        fgets(s, max_line_length, file);
		s[strcspn(s, "\n")] = '\0';
		token = strtok(s, seperate);
		for (k = 0; token != NULL && j < newd->col + newd->str_cols[0]; ) {
			size = (k < newd->str_cols[0]) ? newd->str_cols[k + 1] : -1;
			if (j != size) {
				if (is_nan(token)) num = 0.0f / 0.0f;
				else sscanf(token, "%f", &num);
				newd->data[i][j++] = num;
			} else {
				size = strlen(token);
				newd->str_data[i][k] = (char*)malloc((size >= 3 ? size : 3) + 1);
				if (!is_nan(token)) strcpy(newd->str_data[i][k++], token);
				else strcpy(newd->str_data[i][k++], "nan");
			}
			token = strtok(NULL, seperate);
		}
	}
    free(s);
    fclose(file);
    return newd;
}
void make_csv(char* file_name_csv, Data_Frame* df, char* seperate) {
    FILE* file = fopen(file_name_csv, "w");
	if (!file) {
		printf("Error: Cannot open file !!");
		return ;
	}
	if (!df) return ;
	int i, j;
	if (df->features) {
		for (i = 0; i < df->col; i++) 
			fprintf(file, "%s%s", df->features[i] ? df->features[i] : "null", i == df->col - 1 ? "\n" : seperate);
	}
	for (i = 0; i < df->row; i++) {
		for (j = 0; j < df->col; j++) {
			fprintf(file, "%f%s", df->data[i][j], j == df->col - 1 ? "\n" : seperate);
		}
	}
	fclose(file);
}
void print_data_frame(Data_Frame* df, int col_space, int num_of_rows) {
    if (!df) {
        printf("Error: DataFrame is not existing !!");
        return ;
    }
	if (num_of_rows < 0 || num_of_rows > df->row) num_of_rows = df->row;
    int i, j, k, cur;
	if (df->features) {
		printf("\t");
		j = df->col + df->str_cols[0];
		for (i = 0; i < j; i++) printf("%*s ", col_space, df->features[i]);
		printf("\n");
	}
    for (i = 0; i < num_of_rows; i++) {
        printf("%5d\t", i + 1);
        for (j = 0, k = 0; j < df->col || k < df->str_cols[0]; ) {
			cur = (k < df->str_cols[0]) ? df->str_cols[k + 1] : -1;
			if (j != cur) {
				if (df->data[i][j] == df->data[i][j]) printf("%*.2f ", col_space, df->data[i][j]);
				else printf("%*s ", col_space, "nan");
				j++;
			}
			else printf("%*s ", col_space, df->str_data[i][k++]);
		}
        printf("\n");
    }
}
void free_data_frame(Data_Frame* df) {
	if (!df) return ;
	int i, j;
    if (df->features) {
		for (i = 0; i < df->col; i++) 
			if (df->features[i]) free(df->features[i]);
		free(df->features);
	}
	if (df->data) {
		for (i = 0; i < df->row; i++) 
			if (df->data[i]) free(df->data[i]);
		free(df->data);
	}
	if (df->str_cols[0] > 0) {
		for (i = 0; i < df->row; i++) {
			for (j = 0; j < df->str_cols[0]; j++) free(df->str_data[i][j]);
			free(df->str_data[i]);
		}
		free(df->str_data);
	}
	free(df->str_cols);
    free(df);
}
void describe_df_digit(Data_Frame* df, int col_space) {
	if (!df) {
        printf("Error: DataFrame is not existing !!");
        return ;
    }
    int i, j, k;
	if (df->features) {
		printf("\t");
		for (i = 0, k = 0; i < df->col; i++, k++) {
			for (j = 1; j <= df->str_cols[0]; j++) 
				if (k == df->str_cols[j]) k++;
			printf("%*s ", col_space, df->features[k]);
		}
		printf("\n");
	}
	float** descb = new_matrix(df->col, 5);
	for (i = 0; i < df->col; i++) {
		descb[i][3] = 1e10;
		descb[i][4] = 1e-10;
		for (j = 0; j < df->row; j++) {
			if (df->data[j][i] == df->data[j][i]) {
				descb[i][0]++;
				descb[i][1] += df->data[j][i];
				if (descb[i][3] > df->data[j][i]) descb[i][3] = df->data[j][i];
				if (descb[i][4] < df->data[j][i]) descb[i][4] = df->data[j][i];
			}
		}
		descb[i][1] /= descb[i][0];
		for (j = 0; j < df->row; j++)
			if (df->data[j][i] == df->data[j][i])
				descb[i][2] += (df->data[j][i] - descb[i][1])* (df->data[j][i] - descb[i][1]);
		descb[i][2] /= (descb[i][0] - 1);
		descb[i][2] = (float) sqrt((float) descb[i][2]);
	}
	printf("count\t");
	for (i = 0; i < df->col; i++) printf("%*d ", col_space, (int)descb[i][0]);
	printf("\n mean\t");
	for (i = 0; i < df->col; i++) printf("%*.4f ", col_space, descb[i][1]);
	printf("\n std \t");
	for (i = 0; i < df->col; i++) printf("%*.4f ", col_space, descb[i][2]);
	printf("\n min \t");
	for (i = 0; i < df->col; i++) printf("%*.4f ", col_space, descb[i][3]);
	printf("\n max \t");
	for (i = 0; i < df->col; i++) printf("%*.4f ", col_space, descb[i][4]);
	printf("\n");
	free_matrix(descb, df->col);
}
void describe_df_string(Data_Frame* df, int col_space) {
	if (df->str_cols[0] == 0) {
		printf("DataFrame has no string data !\n");
		return ;
	}
	int i, j, k;
	float** descb = new_matrix(df->str_cols[0], 3);
	char** cdescb = (char**)malloc(df->str_cols[0]* sizeof(char*));
	Set** str_frequence = (Set**)malloc(df->str_cols[0]* sizeof(Set*));
	for (i = 0; i < df->str_cols[0]; i++) {
		str_frequence[i] = new_set(17);
		for (j = 0; j < df->row; j++) {
			if (strcmp(df->str_data[j][i], "nan")) {
				descb[i][0]++;
				if (set_find(str_frequence[i], df->str_data[j][i]))
					(*set_key_access(str_frequence[i], df->str_data[j][i]))++;
				else set_add(str_frequence[i], df->str_data[j][i], 1);
			}
		}
		descb[i][1] = str_frequence[i]->size;
	}
	Space* mid;
	for (k = 0; k < df->str_cols[0]; k++) {
		for (i = 0, j = 0; i < str_frequence[k]->max_size; i++) {
			mid = str_frequence[k]->arr[i];
			while (mid != NULL) {
				if (mid->key > j) {
					j = strlen(mid->keys) + 1;
					cdescb[k] = (char*)malloc(j* sizeof(char));
					strcpy(cdescb[k], mid->keys);
					j = mid->key;
					descb[k][2] = j;
				}
				mid = mid->next;
			}
		}
		free_set(str_frequence[k]);
	}
	free(str_frequence);
	if (df->features) {
		printf("\t");
		for (i = 1; i <= df->str_cols[0]; i++)
			printf("%*s ", col_space, df->features[df->str_cols[i]]);
		printf("\n");
	}
	printf(" count\t");
	for (i = 0 ; i < df->str_cols[0]; i++) printf("%*d ", col_space, (int) descb[i][0]);
	printf("\nunique\t");
	for (i = 0 ; i < df->str_cols[0]; i++) printf("%*d ", col_space, (int) descb[i][1]);
	printf("\n  top \t");
	for (i = 0 ; i < df->str_cols[0]; i++) {
		printf("%*s ", col_space, cdescb[i]);
		free(cdescb[i]);
	}
	printf("\n  fre \t");
	for (i = 0 ; i < df->str_cols[0]; i++) printf("%*d ", col_space, (int) descb[i][2]);
	printf("\n");
	free_matrix(descb, df->str_cols[0]);
	free(cdescb);
}