#include    <math.h>
#include    <string.h>
#include    <unistd.h>
#include    <stdio.h>
#include    <stdlib.h>
#include    <sys/types.h>
#include    <sys/stat.h>
#include    <fcntl.h>
#include    <ctype.h>
#include    <pthread.h>
#include    "HeapSort.cpp"

// Version history


// 8/10/13, 13/10/14, 13/4/18, 6/4/18 (double precision, dim<1000)
// 14/10/20, 20/9/20, 17/1/21
// 24/3/20, 13/4/20, 16/5/20 (VL&VR), 20/9/20
// 22/1/22: allow comma seperating matrix
// 19/4/21: output optimised, diag/rows/cols conversion, $MATRICS_MAXDIM and dim=100 by default

//extern "C" void dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda, double *wr, double *wi, double *vl, int *ldvl, double *vr, int *ldvr, double *work, int *lwork, int *info);
extern "C" void dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda, double *wr, double *wi, double *vl, int *ldvl, double *vr, int *ldvr, double *work, int *lwork, int *info);


bool eigen_decompose(int n, int nv, double * matrix, double * evl_real, double * evl_image, double * ev_left, double * ev_right, double * work, int nwork){
    int ldvl = n; int ldvr = n; int ret = 0;
    //dgeev_("V","V", &n, matrix, &nv, evl_real, evl_image, ev_left, &ldvl, ev_right, &ldvr, work, &nwork, &ret);
    dgeev_("V","V", &n, matrix, &nv, evl_real, evl_image, ev_right, &ldvr, ev_left, &ldvl, work, &nwork, &ret);
    return ret==0;
}
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
const char * software_name = "matrics";
#include    "StringClass.cpp"
#define     MATRIX_DEFAULT_DIM  7
#include    "matrix.cpp"
char * fmt_double   = "%24.16E";
char * fmt_text     = "%24s";
bool wolfram_output = false;
bool tab_output = true;
char * szTitle = "%s matrixA operator [matrix-B [format]]\n";
char * szHelpV = "(c) Cao Siqin 2021.4.19";
char * szHelpB = "\
  Operators:\n\
    = + - * . / // t trace diag Frobenius det dim ev vl vr\n\
    sqrt exp log ln sin cos tan asin acos atan arrange assign\n\
  Operators by element:\n\
    e+ e- e* e. e/ abs sign sgn heaviside\n\
  Statistics:\n\
    sum average stdev max min\n\
  Matrix form:\n\
    file/con/real-number/--../-m..\n\
    -c../-C../-r../-R../-i../-d..\n\
  Format string:\n\
    e.g. -demo/-ds/-dd/-%.15g/%24.18g,/-%%12f/-%%24.16e\n\
    Be cautious when dimension changes\n\
  Maximum dimension: 100 or defined in $MATRICS_MAXDIM\n\
    use export MATRICS_MAXDIM=1000 to extend maximum dimension to 5000\n\
  --help for more details\n\
";
char * szHelpO = "\
  Matrix A and B are N*N\n\
  Maximum dimension: 100 or defined in $MATRICS_MAXDIM\n\
    use export MATRICS_MAXDIM=1000 to extend maximum dimension to 5000\n\
  The support oprators:\n\
    =               show matrix. B is ignored.\n\
    +, -, *         basic calculation\n\
    .               A*B\n\
    /               A*B^-1\n\
    //              A^-1*B\n\
    ..              A^B[1][1], or A^B if B is a real number\n\
    e+, e-, e., e/  calculate by element: Cij = Aij +,-,*,/ Bij\n\
    ^^              direct product\n\
    t               transposition of A. B is ignored\n\
    det, trace, fn, dim |A|, Tr(A), dimension(A). B is ignored.\n\
    inv             A^-1\n\
    ev, vl, vr      eigen decompose (value, left, right) of A\n\
    sin, ...        A*sin(B), ...\n\
      support: abs, sqrt, sgn/sign, step\n\
      support: exp, log, ln, sin, cos, tan, asin, acos, atan\n\
    sum, integrate  sum(A), B ignored\n\
    *2#, *-to-#     diag-to-rows/cols, row1-to-rows/diag, col1-to-cols/diag\n\
    average, stdev  average(A) or [stdev(Aij)]ij, B ignored\n\
    max, min        max{A[][]} or min{A[][]}, B ignored\n\
    arrange/assign  reorder B with A: show B[Aij], index starts from 1\n\
    print/list/show print A with element index (start from 1) in B\n\
    [print-]sub     print sub matrix of A with dimension defined in B\n\
    [print-]diag[onal] print diagonal of matrix A, B ignored\n\
";
char * szHelpM = "\
  The matrix can be:\n\
    file-name       matrix-file, each line contains a matrix\n\
                    will be read twice if dimension not given explicitly\n\
                    avoid to use /dev/stdin if dimension is given\n\
    null, con       the standard input\n\
    a-real-number   a diagonal matrix. 1 for identity matrix\n\
                    a single real number in \"con\" is treated this way\n\
    --.., -m..      a matrix in the matrix-file format\n\
    -c.., -r..      a colume/row vector\n\
    -C.., -R..      a colume/row vector expanding to matrix\n\
    -i..            Aij = A11 matrix\n\
    -d..            an eigenvalue vector represented diagnol matrix\n\
  The data format in the matrix-file:\n\
    sequence: a11 a12 ... a1N a21 a22 ... a2N ... aNN\n\
    ,               csv spacing between elements\n\
    ;               row breaking.\n\
    { } [ ] ( )     ignored.\n\
";
char * szHelpC = "\
  The format string \"-...\" will be recognized as output precision:\n\
    -single         single precision\n\
    -double         double precision\n\
    -demo -ds       single precision with comma=\\n\n\
    -demo-long -dd  (-demo-full) full precision with comma=\\n\n\
    -demo-short     single precision with comma=\\n\n\
    -wolfram        Mathematica note format\n\
    -%12f -%24.16e  manually specify the format. d=demo. Be caucious!\n\
    -%.15g,         high precision and comma separated\n\
    --%.15g         high precision with additional space between rows\n\
    ---%.15g        high precision with line break between rows\n\
";
//
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
void eigen_sort(int n, MatrixNS::Matrix & m, double * ev, MatrixNS::Matrix & m_key, MatrixNS::Matrix & m_tmp){
    double * key = &m_key.a[0][0]; double * index = &m_key.a[1][0];
    for (int i=0; i<n; i++){ key[i] = -ev[i]; index[i] = i; }
    HeapSortNS::HeapSortHeap <double, double> (n, key, index).HeapSort();
    for (int col=0; col<n; col++){
        int m_col = (int)index[col];
        for (int row=0; row<n; row++) m_tmp.a[row][col] = m.a[row][m_col];
    }
    m = m_tmp;
}
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#define CMD_NULL            0
#define CMD_ADD             1
#define CMD_SUB             2
#define CMD_PRODUCT         3
#define CMD_OVER            4
#define CMD_IPRO            5
#define CMD_DIRECTPRO       6
#define CMD_TRANSPOSE       8
#define CMD_DETERMIN        9
#define CMD_TRACE           10
#define CMD_FROBENIUS_NROM  11
#define CMD_F_ABS           101
#define CMD_F_SQRT          102
#define CMD_F_SIN           103
#define CMD_F_COS           104
#define CMD_F_TAN           105
#define CMD_F_EXP           106
#define CMD_F_LOG           107
#define CMD_F_LN            108
#define CMD_F_POW           109
#define CMD_F_ASIN          110
#define CMD_F_ACOS          111
#define CMD_F_ATAN          112
#define CMD_F_SGN           113
#define CMD_F_STEP          114
#define CMD_DIMENSION       115
#define CMD_INVERSE         116
#define CMD_F_ARRANGE       121
#define CMD_EIGENVALUER     122
#define CMD_EIGENVALUEI     123
#define CMD_EIGENVECTORL    124
#define CMD_EIGENVECTORR    125
#define CMD_SUBMATRIX       126
#define CMD_DIAGONAL        127
#define CMD_INTEGRATE       128
#define CMD_DIAG_TO_ROWS    129
#define CMD_DIAG_TO_COLS    130
#define CMD_ROW1_TO_ROWS    131
#define CMD_ROW1_TO_DIAG    132
#define CMD_COL1_TO_COLS    133
#define CMD_COL1_TO_DIAG    134
#define CMD_EADD            21
#define CMD_ESUB            22
#define CMD_EPRODUCT        23
#define CMD_EOVER           24
#define CMD_S_SUM           301
#define CMD_S_AVG           302
#define CMD_S_STDEV         303
#define CMD_S_MAX           304
#define CMD_S_MIN           305
#define CMD_PRINT           401
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#define DEFAULT_MAXDIM 100
size_t max_dim = DEFAULT_MAXDIM;
size_t max_sl = DEFAULT_MAXDIM*DEFAULT_MAXDIM;
size_t max_text = DEFAULT_MAXDIM*DEFAULT_MAXDIM * 24;
size_t get_total_physical_memory(){
    size_t pages = sysconf(_SC_PHYS_PAGES);
    size_t page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}
void set_memory_caps(){
    char * redefined_max_dimension = getenv("MATRICS_MAXDIM");
    if (redefined_max_dimension){
        size_t max_dim_redefined = atol(redefined_max_dimension);
        size_t max_sl_redefined = max_dim_redefined*max_dim_redefined;
        size_t max_text_redefined = max_sl_redefined * 24;
        size_t memory_estimation_min = sizeof(char)*max_text_redefined + sizeof(StringNS::string)*max_sl_redefined + sizeof(double) * 4 * max_dim_redefined;
        size_t memory_estimation_max = memory_estimation_min + (20*sizeof(double))*max_dim_redefined*max_dim_redefined;
        size_t physical_memory = get_total_physical_memory();
        if (memory_estimation_min>physical_memory){
            fprintf(stderr, "%s : fatal error : MATRICS_MAXDIM=%lu requires %g GB, exceeded physical RAM (%g GB)\n", software_name, max_dim_redefined, (double)memory_estimation_min/1024.0/1024.0/1024.0, (double)physical_memory/1024.0/1024.0/1024.0);
            exit(-1);
        } else if (memory_estimation_max>physical_memory){
            fprintf(stderr, "%s : warning : MATRICS_MAXDIM=%lu (%g GB) may exceed physical RAM (%g GB)\n", software_name, max_dim_redefined, (double)memory_estimation_max/1024.0/1024.0/1024.0, (double)physical_memory/1024.0/1024.0/1024.0);
        } else {
            max_dim = max_dim_redefined;
            max_sl = max_sl_redefined;
            max_text = max_text_redefined;
        }
    }
}
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
void dispose_all(MatrixNS::Matrix * ma, MatrixNS::Matrix * mb, MatrixNS::Matrix * mc, MatrixNS::Matrix * md, MatrixNS::Matrix * me, FILE * fa, FILE * fb){
    if(ma) ma->dispose(); if (mb) mb->dispose(); if (mc) mc->dispose(); if (md) md->dispose(); if (me) me->dispose(); if (fa) fclose(fa); if (fb) fclose(fb);
}

//char input[max_text];
//StringNS::string sl[max_sl];
char * input = NULL;
StringNS::string * sl = NULL;
int read_matrix(StringNS::string sline, MatrixNS::Matrix * m, int N, bool analysis_matrix=true){
    if (m) *m = 0;
    for (int i=0; i<sline.length; i++){
        char c = sline.text[i]; if (c=='{' || c=='}' || c=='(' || c==')' || c=='[' || c==']') sline.text[i] = ' ';
    }

    bool csv_line = false; for (int i=0; i<sline.length && !csv_line; i++) if (sline[i]==',') csv_line = true;
    int nw = 0;
    if (csv_line){
        nw = StringNS::analysis_csv_line(sline, sl, max_sl, true, true);
    } else {
        nw = StringNS::analysis_line(sline, sl, max_sl, true);
    }

    int x = 0; int y = 0; int nele = 0; int maxx = 0; bool newline = false; bool use_x_y_range = false;
    for (int i=0; i<nw; i++){
        char c = sl[i].text[0];
        if (c=='r' || c=='n' || c=='\r' || c=='\n'){ // short format, end a line and start a new line immediately
            if (!newline){
                x = 0; y ++; newline = false;
            }
            use_x_y_range = true;
        } else if (x>=0 && x<N && y>=0 && y<N){
            if (m) *m->e(y,x) = atof(sl[i].text);
            x ++;
            if (x>=N){ x = 0; y ++; newline = true; } else newline = false;
            nele ++;
        }
        if (y > N) break;
        if (x>maxx) maxx = x;
    }

    if (x>maxx) maxx = x;
    if (analysis_matrix){
        int nele2 = (y+1)*(y+1);
        int nele3 = (maxx)*(maxx);
        if (use_x_y_range && nele<nele2) nele = nele2;
        if (use_x_y_range && nele<nele3) nele = nele3;
    }

    if (nele==1 && m){
        for (int i=1; i<m->n; i++) *m->e(i, i) = *m->e(0, 0);
    }

    return nele;
}
int read_matrix_file(FILE * file, MatrixNS::Matrix * m, int N){
    bool ret = fgets(input, max_text, file);
    int nele = read_matrix(input, m, N);
    return ret? nele : -1;
}
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
double sgn(double a){ return a>0? 1 : a<0? -1 : 0; }
double step(double a){ return 0.5 * (1 + sgn(a)); }
void rearrange(MatrixNS::Matrix * m, MatrixNS::Matrix * idx, MatrixNS::Matrix * tmp){
    for (int x=0; x<m->n; x++) for (int y=0; y<m->n; y++){
        int ii = (int)*idx->e(y,x) - 1;
        if (ii>=0 && ii<m->n*m->n){
            int i = ii % m->n; int j = ii / m->n;
            *tmp->e(y,x) = *m->e(j,i);
        } else *tmp->e(y,x) = 0;
    }
}


void print_matrix_data(double * m, int dim, const char * sep, const char * format="%9.3g"){
    if (!format||!format[0]) format = "%9.3g";
    for (int a=0; a<dim; a++){
        for (int b=0; b<dim; b++){
            printf(format, fabs(m[a*dim+b])<1e-99?0:m[a*dim+b]);
            printf(" ");
        }; printf("%s", sep);
    }
    printf("\n");
}
typedef void (FunctionOfMatrix) (int,double*,double param);
void matrix_ln(int dim, double * data, double param){
    double loge = log(2.7182818284590452353602874713527);
    for (int i=0; i<dim; i++) data[i*dim+i] = log(data[i*dim+i])/loge;
}
void matrix_lg(int dim, double * data, double param){
    double loge = log(10.0);
    for (int i=0; i<dim; i++) data[i*dim+i] = log(data[i*dim+i])/loge;
}
void matrix_exp(int dim, double * data, double param){
    for (int i=0; i<dim; i++) data[i*dim+i] = exp(data[i*dim+i]);
}

void matrix_sqrt(int dim, double * data, double param){
    for (int i=0; i<dim; i++) data[i*dim+i] = sqrt(data[i*dim+i]);
}
void matrix_pow(int dim, double * data, double param){
    for (int i=0; i<dim; i++) data[i*dim+i] = pow(data[i*dim+i], param);
}
void matrix_sin(int dim, double * data, double param){ for (int i=0; i<dim; i++) data[i*dim+i] = sin(data[i*dim+i]); }
void matrix_cos(int dim, double * data, double param){ for (int i=0; i<dim; i++) data[i*dim+i] = cos(data[i*dim+i]); }
void matrix_tan(int dim, double * data, double param){ for (int i=0; i<dim; i++) data[i*dim+i] = tan(data[i*dim+i]); }
void matrix_asin(int dim, double * data, double param){ for (int i=0; i<dim; i++) data[i*dim+i] = asin(data[i*dim+i]); }
void matrix_acos(int dim, double * data, double param){ for (int i=0; i<dim; i++) data[i*dim+i] = acos(data[i*dim+i]); }
void matrix_atan(int dim, double * data, double param){ for (int i=0; i<dim; i++) data[i*dim+i] = atan(data[i*dim+i]); }

double Frobenius_norm(MatrixNS::Matrix & a){
    double sum = 0;
    for (int i=0; i<a.n*a.n; i++) sum += a.m[i] * a.m[i];
    return sqrt(fabs(sum));
}

MatrixNS::Matrix perform_matrix_function(MatrixNS::Matrix & A, MatrixNS::Matrix & ret, MatrixNS::Matrix * temp[6], double * eigen_work, FunctionOfMatrix * matrix_function, double param=0){ // perform A * f(A)
    memset(eigen_work, 0, sizeof(double) * 4 * max_dim);
    MatrixNS::Matrix * ev = temp[0];
    MatrixNS::Matrix * evi = temp[1];
    MatrixNS::Matrix * vlt = temp[2];
    MatrixNS::Matrix * vr = temp[3];
    ret = A;
  // eigen decomposition first
    //eigen_decompose(A.n, A.n, &ret.a[0][0], &ev->a[0][0], &evi->a[0][0], &vr->a[0][0], &vlt->a[0][0], eigen_work, 4*max_dim);
    eigen_decompose(A.n, A.n, &ret.a[0][0], &ev->a[0][0], &evi->a[0][0], &vlt->a[0][0], &vr->a[0][0], eigen_work, 4*max_dim);
    vr->Transpose();
  // convert eigenvalue array into eigenvalue matrix
    for (int i=1; i<A.n; i++){
        ev->a[i][i] = ev->a[0][i]; ev->a[0][i] = 0;
        evi->a[i][i] = ev->a[0][i]; evi->a[0][i] = 0;
    }

//    printf("A:\n"); print_matrix_data(A.m, A.n, "\n");
//    printf("ev:\n"); print_matrix_data(ev->m, A.n, "\n");
//    printf("vlt:\n"); print_matrix_data(vlt->m, A.n, "\n");
//    printf("vr:\n"); print_matrix_data(vr->m, A.n, "\n");

  // get normalized eigenvectors
    *temp[4] = 0; for (int i=0; i<A.n; i++){
        double rescale = 0; for (int j=0; j<A.n; j++) rescale += vlt->a[i][j] * vr->a[j][i];
        temp[4]->a[i][i] = sqrt(fabs(1/rescale));
    }
    * vlt = *temp[4] * *vlt; * vr = *vr * *temp[4];

  // do function
    matrix_function(A.n, ev->m, param);

    //ret = *vlt * A; ret = ret * *vr; // test L^T*A*V = lambda
    //ret = *vr * *ev; ret = ret * *vlt; // test V*lambda*L^T = A

  // reconstructing the result matrix
    ret = *vr * *ev; ret = ret * *vlt;

    //printf("vlt:\n"); print_matrix_data(vlt->m, A.n, "\n");
    //printf("vr:\n"); print_matrix_data(vr->m, A.n, "\n");
    //printf("ret:\n"); print_matrix_data(ret.m, A.n, "\n");

    return ret;
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
int main(int argc, char *argv[]){

    set_memory_caps();

    input = (char*)malloc(sizeof(char)*max_text);
    sl = (StringNS::string*)malloc(sizeof(StringNS::string)*max_sl);
    if (!input || !sl){ fprintf(stderr, "malloc failure\n"); return -1; }

    char * comma = ""; char * comma_arg = "";
    char fmt_buffer_double[32]; strcpy(fmt_buffer_double, fmt_double); fmt_double = fmt_buffer_double;
    char fmt_buffer_text[32]; strcpy(fmt_buffer_text, fmt_text); fmt_text = fmt_buffer_text;

    if (argc<=1){
        printf(szTitle, software_name);
        return 0;
    } else if (StringNS::string(argv[1])=="-h" || StringNS::string(argv[1])=="--h"){
        printf(szTitle, software_name);
        printf("  %s\n", szHelpV);
        printf("%s", szHelpB);
        return 0;
    } else if (StringNS::string(argv[1])=="-help" || StringNS::string(argv[1])=="--help"){
        printf(szTitle, software_name);
        printf("  %s\n", szHelpV);
        printf("%s", szHelpO);
        printf("%s", szHelpM);
        printf("%s", szHelpC);
        return 0;
    }

    int argv_nbase = 0;
    int N = 0;

  // determine the matrix dimenstion
    if (argv[1][0] == '-' && (argv[1][1] == '-'||argv[1][1] == 'm'||argv[1][1] == 'M')){
        strcpy(input, &argv[1][2]);
        int nw = read_matrix(input, null, max_sl);
        N = (int) sqrt((double) nw); if (N*N<nw) N++;
        argv_nbase = -1;
    } else if (argv[1][0] == '-' && ((argv[1][1] == 'c'||argv[1][1] == 'C') || (argv[1][1] == 'r'||argv[1][1] == 'R') || (argv[1][1] == 'd'||argv[1][1] == 'D'))){
        strcpy(input, &argv[1][2]);
        int nw = read_matrix(input, null, max_sl, false);
        N = nw;
        argv_nbase = -1;
    } else if (argv[1][0] == '-' && (argv[1][1] == 'i'||argv[1][1] == 'I')){
        fprintf(stderr, "%s : error : cannot define matrix dimension by special matrix\n", software_name); return -1;
    } else if (!StringNS::is_string_number(argv[1])){
        FILE * ft = fopen(argv[1], "r"); if (ft){
            int nw = read_matrix_file(ft, null, max_sl);
            N = (int) sqrt((double) nw); if (N*N<nw) N++;
            fclose(ft);
            argv_nbase = -1;
        } else { fprintf(stderr, "%s : error : can't open file %s\n", software_name, argv[1]); return -1; }
    } else {
        N = atoi(argv[argv_nbase+1]);
    }
    if (N<=0 || N>max_dim){ fprintf(stderr, "%s : error : dimension (%d) too %s\n", software_name, N, N<=0?"small":"big"); return -1; }

  // compile operator
    char * szcmd = (argv_nbase+3<argc) ? argv[argv_nbase+3] : (char*)"=";
    int command = 0;
    if (StringNS::string(szcmd) == "+"){                command = CMD_ADD           ;
    } else if (StringNS::string(szcmd) == "-"){         command = CMD_SUB           ;
    } else if (StringNS::string(szcmd) == "."){         command = CMD_PRODUCT       ;
    } else if (StringNS::string(szcmd) == "*"){         command = CMD_PRODUCT       ;
    } else if (StringNS::string(szcmd) == "/"){         command = CMD_OVER          ;
    } else if (StringNS::string(szcmd) == "//"){        command = CMD_IPRO          ;
    } else if (StringNS::string(szcmd) == "^^"){        command = CMD_DIRECTPRO     ;
    } else if (StringNS::string(szcmd) == "t"){         command = CMD_TRANSPOSE     ;
    } else if (StringNS::string(szcmd) == "det"){       command = CMD_DETERMIN      ;
    } else if (StringNS::string(szcmd) == "tr"){        command = CMD_TRACE         ;
    } else if (StringNS::string(szcmd) == "fn"){        command = CMD_FROBENIUS_NROM;
    } else if (StringNS::string(szcmd) == "Frobenius-norm"){ command = CMD_FROBENIUS_NROM;
    } else if (StringNS::string(szcmd) == "Frobenius_norm"){ command = CMD_FROBENIUS_NROM;
    } else if (StringNS::string(szcmd) == "Frobenius"){ command = CMD_FROBENIUS_NROM;
    } else if (StringNS::string(szcmd) == "trace"){     command = CMD_TRACE         ;
    } else if (StringNS::string(szcmd) == "abs"){       command = CMD_F_ABS         ;
    } else if (StringNS::string(szcmd) == "sqrt"){      command = CMD_F_SQRT        ;
    } else if (StringNS::string(szcmd) == "sin"){       command = CMD_F_SIN         ;
    } else if (StringNS::string(szcmd) == "cos"){       command = CMD_F_COS         ;
    } else if (StringNS::string(szcmd) == "tan"){       command = CMD_F_TAN         ;
    } else if (StringNS::string(szcmd) == "exp"){       command = CMD_F_EXP         ;
    } else if (StringNS::string(szcmd) == "log"){       command = CMD_F_LOG         ;
    } else if (StringNS::string(szcmd) == "ln"){        command = CMD_F_LN          ;
    } else if (StringNS::string(szcmd) == "pow"){       command = CMD_F_POW         ;
    } else if (StringNS::string(szcmd) == "^"){         command = CMD_F_POW         ;
    } else if (StringNS::string(szcmd) == ".."){        command = CMD_F_POW         ;
    } else if (StringNS::string(szcmd) == "**"){        command = CMD_F_POW         ;
    } else if (StringNS::string(szcmd) == "asin"){      command = CMD_F_ASIN        ;
    } else if (StringNS::string(szcmd) == "acos"){      command = CMD_F_ACOS        ;
    } else if (StringNS::string(szcmd) == "atan"){      command = CMD_F_ATAN        ;
    } else if (StringNS::string(szcmd) == "assign"){    command = CMD_F_ARRANGE     ;
    } else if (StringNS::string(szcmd) == "arrange"){   command = CMD_F_ARRANGE     ;
    } else if (StringNS::string(szcmd) == "rearrange"){ command = CMD_F_ARRANGE     ;
    } else if (StringNS::string(szcmd) == "sgn"){       command = CMD_F_SGN         ;
    } else if (StringNS::string(szcmd) == "sign"){      command = CMD_F_SGN         ;
    } else if (StringNS::string(szcmd) == "step"){      command = CMD_F_STEP        ;
    } else if (StringNS::string(szcmd) == "heaviside"){ command = CMD_F_STEP        ;
    } else if (StringNS::string(szcmd) == "dim"){       command = CMD_DIMENSION     ;
    } else if (StringNS::string(szcmd) == "inv"){       command = CMD_INVERSE       ;
    } else if (StringNS::string(szcmd) == "ev"){        command = CMD_EIGENVALUER   ;
    } else if (StringNS::string(szcmd) == "evr"){       command = CMD_EIGENVALUER   ;
    } else if (StringNS::string(szcmd) == "evi"){       command = CMD_EIGENVALUEI   ;
    } else if (StringNS::string(szcmd) == "el"){        command = CMD_EIGENVECTORL  ;
    } else if (StringNS::string(szcmd) == "vl"){        command = CMD_EIGENVECTORL  ;
    } else if (StringNS::string(szcmd) == "er"){        command = CMD_EIGENVECTORR  ;
    } else if (StringNS::string(szcmd) == "vr"){        command = CMD_EIGENVECTORR  ;
    } else if (StringNS::string(szcmd) == "e+"){        command = CMD_EADD          ;
    } else if (StringNS::string(szcmd) == "e-"){        command = CMD_ESUB          ;
    } else if (StringNS::string(szcmd) == "e*"){        command = CMD_EPRODUCT      ;
    } else if (StringNS::string(szcmd) == "e."){        command = CMD_EPRODUCT      ;
    } else if (StringNS::string(szcmd) == "e/"){        command = CMD_EOVER         ;
    } else if (StringNS::string(szcmd) == "="){         command = CMD_NULL          ;
    } else if (StringNS::string(szcmd) == "sum"){       command = CMD_S_SUM         ;
    } else if (StringNS::string(szcmd) == "in"){        command = CMD_INTEGRATE     ;
    } else if (StringNS::string(szcmd) == "integrate"){ command = CMD_INTEGRATE     ;
    } else if (StringNS::string(szcmd) == "integral"){  command = CMD_INTEGRATE     ;

    } else if (StringNS::string(szcmd) == "diag2rows"){ command = CMD_DIAG_TO_ROWS  ;
    } else if (StringNS::string(szcmd) == "diag2cols"){ command = CMD_DIAG_TO_COLS  ;
    } else if (StringNS::string(szcmd) == "row12rows"){ command = CMD_ROW1_TO_ROWS  ;
    } else if (StringNS::string(szcmd) == "row12diag"){ command = CMD_ROW1_TO_DIAG  ;
    } else if (StringNS::string(szcmd) == "col12cols"){ command = CMD_COL1_TO_COLS  ;
    } else if (StringNS::string(szcmd) == "col12diag"){ command = CMD_COL1_TO_DIAG  ;
    } else if (StringNS::string(szcmd) == "diagtorows"){ command = CMD_DIAG_TO_ROWS  ;
    } else if (StringNS::string(szcmd) == "diagtocols"){ command = CMD_DIAG_TO_COLS  ;
    } else if (StringNS::string(szcmd) == "row1torows"){ command = CMD_ROW1_TO_ROWS  ;
    } else if (StringNS::string(szcmd) == "row1todiag"){ command = CMD_ROW1_TO_DIAG  ;
    } else if (StringNS::string(szcmd) == "col1tocols"){ command = CMD_COL1_TO_COLS  ;
    } else if (StringNS::string(szcmd) == "col1todiag"){ command = CMD_COL1_TO_DIAG  ;
    } else if (StringNS::string(szcmd) == "diag-to-rows"){ command = CMD_DIAG_TO_ROWS  ;
    } else if (StringNS::string(szcmd) == "diag-to-cols"){ command = CMD_DIAG_TO_COLS  ;
    } else if (StringNS::string(szcmd) == "row1-to-rows"){ command = CMD_ROW1_TO_ROWS  ;
    } else if (StringNS::string(szcmd) == "row1-to-diag"){ command = CMD_ROW1_TO_DIAG  ;
    } else if (StringNS::string(szcmd) == "col1-to-cols"){ command = CMD_COL1_TO_COLS  ;
    } else if (StringNS::string(szcmd) == "col1-to-diag"){ command = CMD_COL1_TO_DIAG  ;

    } else if (StringNS::string(szcmd) == "mean"){      command = CMD_S_AVG         ;
    } else if (StringNS::string(szcmd) == "avg"){       command = CMD_S_AVG         ;
    } else if (StringNS::string(szcmd) == "average"){   command = CMD_S_AVG         ;
    } else if (StringNS::string(szcmd) == "stdev"){     command = CMD_S_STDEV       ;
    } else if (StringNS::string(szcmd) == "stderr"){    command = CMD_S_STDEV       ;
    } else if (StringNS::string(szcmd) == "max"){       command = CMD_S_MAX         ;
    } else if (StringNS::string(szcmd) == "min"){       command = CMD_S_MIN         ;
    } else if (StringNS::string(szcmd) == "print"){     command = CMD_PRINT         ;
    } else if (StringNS::string(szcmd) == "list"){      command = CMD_PRINT         ;
    } else if (StringNS::string(szcmd) == "show"){      command = CMD_PRINT         ;
    } else if (StringNS::string(szcmd) == "print-sub"){ command = CMD_SUBMATRIX     ;
    } else if (StringNS::string(szcmd) == "print_sub"){ command = CMD_SUBMATRIX     ;
    } else if (StringNS::string(szcmd) == "sub"){       command = CMD_SUBMATRIX     ;
    } else if (StringNS::string(szcmd) == "print-diag"){ command = CMD_DIAGONAL     ;
    } else if (StringNS::string(szcmd) == "print-diagonal"){ command = CMD_DIAGONAL     ;
    } else if (StringNS::string(szcmd) == "diag"){      command = CMD_DIAGONAL      ;
    } else if (StringNS::string(szcmd) == "diagonal"){  command = CMD_DIAGONAL      ;
    } else if (szcmd[0]=='-'){                          comma_arg = szcmd;
    } else { fprintf(stderr, "%s : error : unknown operator %s\n", software_name, szcmd); return -1; }
    int output_type = 0;
    if (command==CMD_DETERMIN||command==CMD_TRACE||command==CMD_DIMENSION||command==CMD_FROBENIUS_NROM) output_type=2;
    if (command==CMD_S_SUM||command==CMD_S_AVG||command==CMD_S_STDEV ||command==CMD_S_MAX||command==CMD_S_MIN) output_type=3;
    if (command==CMD_DIMENSION) fmt_double = "%.0f";
    if (command==CMD_PRINT || command==CMD_SUBMATRIX || command==CMD_DIAGONAL) output_type = 4;

    MatrixNS::MatrixDataContainer mcontainer; mcontainer.init(10*max_dim*max_dim);
    MatrixNS::Matrix A, B, C, Iv, T; double value = 0;
    MatrixNS::Matrix MS, MSS, MESS, MaxS, MinS; int data_count = 0;
    A.init(&mcontainer, N); B.init(&mcontainer, N); Iv.init(&mcontainer, N); T.init(&mcontainer, N);
    MS.init(&mcontainer, N); MSS.init(&mcontainer, N); MESS.init(&mcontainer, N); MaxS.init(&mcontainer, N); MinS.init(&mcontainer, N);
    if (command==CMD_DIRECTPRO) C.init(&mcontainer, N*N); else C.init(&mcontainer, N);
    MS = 0; MSS = 0; MESS = 0; MaxS = 0; MinS = 0; bool MaxMinSSet = false;

    double * eigen_work = (double*)malloc(sizeof(double) * 4 * max_dim);

/*
    double * eigen_values[2];
        eigen_values[0] = (double*) malloc(sizeof(double)*N*N);
        eigen_values[1] = (double*) malloc(sizeof(double)*N*N);
    double * eigen_vectors[2];
        eigen_vectors[0] = (double*) malloc(sizeof(double)*N*N);
        eigen_vectors[1] = (double*) malloc(sizeof(double)*N*N);
*/

  // initialise constant matirx
    MatrixNS::Matrix * maddr[2] = { &A, &B };
    char * szfile[2]; FILE * file[2] = { null, null }; bool check[2] = { true, true }; bool omit_m_B = false;
    szfile[0] = argv[argv_nbase+2]; szfile[1] = "1"; if (argc>argv_nbase+4) szfile[1] = argv[argv_nbase+4];
    for (int i=0; i<2; i++){
        if (szfile[i][0] == '-' && (szfile[i][1] == '-'||szfile[i][1] == 'm'||szfile[i][1] == 'M')){
            strcpy(input, &szfile[i][2]);
            int nw = read_matrix(input, maddr[i], N); check[i] = false;
        } else if (szfile[i][0] == '-' && (szfile[i][1] == 'c'||szfile[i][1] == 'C')){
            strcpy(input, &szfile[i][2]);
            int nw = read_matrix(input, maddr[i], N, false); check[i] = false;
            if (szfile[i][1] == 'C') for (int x=1; x<N; x++) for (int y=0; y<N; y++) *maddr[i]->e(x,y) = *maddr[i]->e(0,y);
            maddr[i]->Transpose();
        } else if (szfile[i][0] == '-' && (szfile[i][1] == 'r'||szfile[i][1] == 'R')){
            strcpy(input, &szfile[i][2]);
            int nw = read_matrix(input, maddr[i], N, false); check[i] = false;
            if (szfile[i][1] == 'R') for (int x=1; x<N; x++) for (int y=0; y<N; y++) *maddr[i]->e(x,y) = *maddr[i]->e(0,y);
        } else if (szfile[i][0] == '-' && (szfile[i][1] == 'd'||szfile[i][1] == 'D')){
            strcpy(input, &szfile[i][2]);
            int nw = read_matrix(input, maddr[i], N, false); check[i] = false;
            for (int x=1; x<N; x++) {
                *maddr[i]->e(x,x) = *maddr[i]->e(0,x);
                if (x != 0){ *maddr[i]->e(0,x) = 0; }
            }
        } else if (szfile[i][0] == '-' && (szfile[i][1] == 'i'||szfile[i][1] == 'I')){
            strcpy(input, &szfile[i][2]);
            int nw = read_matrix(input, maddr[i], N, false); check[i] = false;
            for (int x=0; x<N; x++) for (int y=0; y<N; y++) if (x!=0 || y!=0) *maddr[i]->e(x,y) = *maddr[i]->e(0,0);
        } else if (is_string_number(StringNS::string(szfile[i]))){
            *maddr[i] = atof(szfile[i]); check[i] = false;
        } else if (StringNS::string(szfile[i])=="null" || StringNS::string(szfile[i])=="con"){
            file[i] = stdin;
        } else {
            file[i] = fopen(szfile[i], "r");
        }
    }
    int omit_B_list[] = { CMD_NULL, CMD_TRANSPOSE, CMD_DETERMIN, CMD_TRACE, CMD_FROBENIUS_NROM, CMD_DIMENSION, CMD_INVERSE, CMD_S_SUM, CMD_INTEGRATE, CMD_DIAG_TO_ROWS,CMD_DIAG_TO_COLS,CMD_ROW1_TO_ROWS,CMD_ROW1_TO_DIAG,CMD_COL1_TO_COLS,CMD_COL1_TO_DIAG, CMD_S_AVG, CMD_S_STDEV , CMD_S_MAX, CMD_S_MIN, CMD_EIGENVALUER, CMD_EIGENVALUEI, CMD_EIGENVECTORL, CMD_EIGENVECTORR, CMD_DIAGONAL };
    for (int i=0; i<sizeof(omit_B_list)/sizeof(omit_B_list[0]); i++) if (command==omit_B_list[i]){
        check[1] = false; omit_m_B = true;
    }

    if ((check[0] && !file[0]) || (check[1] && !file[1])){
        fprintf(stderr, "%s : error : can't open file %s%s%s\n", software_name, (check[0] && !file[0])?szfile[0]:szfile[1], ((check[0] && !file[0])&&(check[1] && !file[1]))?" and ":"", ((check[0] && !file[0])&&(check[1] && !file[1]))?szfile[1]:"");
        dispose_all(&A, &B, &C, &Iv, &T, file[0], file[1]);
        dispose_all(&MS, &MSS, &MESS, &MaxS, &MinS, null, null);
        mcontainer.dispose();
        return -1;
    }

    if (!comma_arg || !comma_arg[0]){
        if (argc>argv_nbase+5) comma_arg = argv[argv_nbase+5];
        if (omit_m_B && argc>argv_nbase+4) comma_arg = argv[argv_nbase+4];
    }
    if (comma_arg[0] == '-'){
        bool enforce_tab = false; bool enforce_demo = false;
        if (comma_arg[0] == '-') comma_arg = &comma_arg[1];
        if (comma_arg[0] == '-'){ comma_arg = &comma_arg[1]; enforce_tab = true; }
        if (comma_arg[0] == '-'){ comma_arg = &comma_arg[1]; enforce_demo = true; }

        if (StringNS::string(comma_arg) == "single"){
            fmt_double = "%12g"; fmt_text = "%12s";
        } else if (StringNS::string(comma_arg) == "double"){
            fmt_double = "%24.16g"; fmt_text = "%24s";
        } else if (StringNS::string(comma_arg) == "demo"){
            fmt_double = "%12g"; fmt_text = "%12s"; comma = "\n";
        } else if (StringNS::string(comma_arg) == "ds" || StringNS::string(comma_arg) == "demo-single" || StringNS::string(comma_arg) == "demo-short"){
            fmt_double = "%12g"; fmt_text = "%12s"; comma = "\n";
        } else if (StringNS::string(comma_arg) == "df" || StringNS::string(comma_arg) == "dl" || StringNS::string(comma_arg) == "dd" || StringNS::string(comma_arg) == "demo-full" || StringNS::string(comma_arg) == "demo-double" || StringNS::string(comma_arg) == "demo-long"){
            fmt_double = "%24.16g"; fmt_text = "%12s"; comma = "\n";
        } else if (StringNS::string(comma_arg) == "wolfram"){
            fmt_double = "%.16g"; fmt_text = "%12s"; wolfram_output = true;
        } else {
            strcpy(fmt_buffer_double, comma_arg); fmt_double = fmt_buffer_double;
            strcpy(fmt_buffer_text, comma_arg); fmt_text = fmt_buffer_text;
            for (int i=0; i<sizeof(fmt_buffer_text) && fmt_buffer_text[i]; i++) if (fmt_buffer_text[i]=='f'||fmt_buffer_text[i]=='F'||fmt_buffer_text[i]=='e'||fmt_buffer_text[i]=='E'||fmt_buffer_text[i]=='d'||fmt_buffer_text[i]=='D') fmt_buffer_text[i]= 's';
            for (int i=0; i<sizeof(fmt_buffer_double) && fmt_buffer_double[i]; i++) if (fmt_buffer_double[i]=='d'||fmt_buffer_double[i]=='D'){ fmt_buffer_double[i]='g'; comma = "\n"; }
        }
        if (wolfram_output){
            tab_output = false;
        } else {
            for (int j=0; j<sizeof(fmt_double)&&fmt_double[j]; j++){
                if (fmt_double[j]==',' || fmt_double[j]==' ' || fmt_double[j]=='\t' || fmt_double[j]=='\n' || fmt_double[j]==';'){
                    tab_output = false;
                }
            }
        }

        if (enforce_demo){
            comma = "\n";
        } else if (enforce_tab){
            if (tab_output && !comma[0]) comma = "  ";
            else comma = " ";
        } else {
            if (tab_output && !comma[0]) comma = " ";
        }
    }

  // handle all matrices
    bool success[2] = { true, true };
    double lg10 = log(10.0); double lge = log(2.71828182845904523536028747135);
    mcontainer.save_ip();
    while (success[0] && success[1]){
        mcontainer.restore_ip();
        mcontainer.save_ip();

        MatrixNS::Matrix * matrix_function_param_array[6] = { &MS, &Iv, &MSS, &MESS, &MaxS, &MinS };

        bool fail_inverse = false;
        if (file[0]) if(read_matrix_file(file[0], &A, N)<0) success[0] = false;
        if (file[1]){
          if (file[0] == file[1]){
            B = A; success[1] = success[0];
          } else {
            if(read_matrix_file(file[1], &B, N)<0) success[1] = false;
          }
        }
        if (!success[0] || !success[1]) break;

        if (output_type==3){
            data_count++;
            MS += A; MSS += A*A; for (int i=0; i<A.n; i++) for (int j=0; j<A.n; j++) *MESS.e(i,j) += *A.e(i,j) * *A.e(i,j);
            if (!MaxMinSSet){ MaxMinSSet = true; MaxS = A; MinS = A;
            } else {
                for (int i=0; i<A.n; i++) for (int j=0; j<A.n; j++){
                    if (A.a[i][j] > MaxS.a[i][j]) MaxS.a[i][j] = A.a[i][j];
                    if (A.a[i][j] < MinS.a[i][j]) MinS.a[i][j] = A.a[i][j];
                }
            }
        }

        switch (command){
        case CMD_NULL       : C = A; break;
        case CMD_DIRECTPRO  : {
            for (int y=0; y<C.n; y++) for (int x=0; x<C.n; x++) *C.e(y, x) = 0;
            for (int y1=0; y1<N; y1++) for (int x1=0; x1<N; x1++) for (int y2=0; y2<N; y2++) for (int x2=0; x2<N; x2++){
                int k = y1*N*N*N + x1*N + y2*N*N + x2;
                *C.e(k/C.n, k%C.n) = *A.e(y1, x1) * *B.e(y2, x2);
            }
          }; break;
        case CMD_ADD        : C = A; C += B; break;
        case CMD_SUB        : C = A; C -= B; break;
        case CMD_PRODUCT    : C = A * B; break;
        case CMD_OVER       : if (!B.inverse(Iv, T)) fail_inverse |= true; C = A * Iv; break;
        case CMD_IPRO       : if (!A.inverse(Iv, T)) fail_inverse |= true; C = Iv * B; break;
        case CMD_TRANSPOSE  : C = A; C.Transpose(); break;
        case CMD_DETERMIN   : {
            value = A.determin();
          } break;
        case CMD_TRACE      : value = A.trace(); break;
        case CMD_FROBENIUS_NROM : value = Frobenius_norm(A); break;
        case CMD_F_ABS      : for (int y=0; y<N; y++) for (int x=0; x<N; x++) *B.e(y,x) = fabs(*B.e(y,x)); C = A * B; break;
        case CMD_F_SQRT     : C = A * perform_matrix_function(B, C, matrix_function_param_array, eigen_work, matrix_sqrt); break;
        case CMD_F_SIN      : C = A * perform_matrix_function(B, C, matrix_function_param_array, eigen_work, matrix_sin); break;
        case CMD_F_COS      : C = A * perform_matrix_function(B, C, matrix_function_param_array, eigen_work, matrix_cos); break;
        case CMD_F_TAN      : C = A * perform_matrix_function(B, C, matrix_function_param_array, eigen_work, matrix_tan); break;
        case CMD_F_EXP      : C = A * perform_matrix_function(B, C, matrix_function_param_array, eigen_work, matrix_exp); break;
        case CMD_F_LOG      : C = A * perform_matrix_function(B, C, matrix_function_param_array, eigen_work, matrix_lg); break;
        case CMD_F_LN       : C = A * perform_matrix_function(B, C, matrix_function_param_array, eigen_work, matrix_ln); break;
        //case CMD_F_POW      : C = perform_matrix_function(A, C, matrix_function_param_array, eigen_work, matrix_pow, B.trace()/B.n); break;
        case CMD_F_POW      : C = perform_matrix_function(A, C, matrix_function_param_array, eigen_work, matrix_pow, B.a[0][0]); break;
        case CMD_F_ASIN     : C = A * perform_matrix_function(B, C, matrix_function_param_array, eigen_work, matrix_asin); break;
        case CMD_F_ACOS     : C = A * perform_matrix_function(B, C, matrix_function_param_array, eigen_work, matrix_acos); break;
        case CMD_F_ATAN     : C = A * perform_matrix_function(B, C, matrix_function_param_array, eigen_work, matrix_atan); break;
        case CMD_F_SGN      : for (int y=0; y<N; y++) for (int x=0; x<N; x++) *B.e(y,x) = sgn(*B.e(y,x)); C = A * B; break;
        case CMD_F_STEP     : for (int y=0; y<N; y++) for (int x=0; x<N; x++) *B.e(y,x) = step(*B.e(y,x)); C = A * B; break;
        case CMD_DIMENSION  : value = N; break;
        case CMD_INVERSE    : if (!A.inverse(C, T)) fail_inverse |= true; break;
        case CMD_EIGENVALUER:
            B = C = Iv = MESS = MSS = 0; memset(eigen_work, 0, sizeof(double) * 4 * max_dim);
            eigen_decompose(N, N, &A.a[0][0], &C.a[0][0], &Iv.a[0][0], &MESS.a[0][0], &MSS.a[0][0], eigen_work, 4*max_dim);
            eigen_sort(N, C, &C.a[0][0], MaxS, MinS);
            for (int i=1; i<C.n; i++){ C.a[i][i] = C.a[0][i]; C.a[0][i] = 0; }
            break;
        case CMD_EIGENVALUEI:
            B = C = Iv = MESS = MSS = 0; memset(eigen_work, 0, sizeof(double) * 4 * max_dim);
            eigen_decompose(N, N, &A.a[0][0], &Iv.a[0][0], &C.a[0][0], &MESS.a[0][0], &MSS.a[0][0], eigen_work, 4*max_dim);
            eigen_sort(N, C, &Iv.a[0][0], MaxS, MinS);
            for (int i=1; i<C.n; i++){ C.a[i][i] = C.a[0][i]; C.a[0][i] = 0; }
            break;
        case CMD_EIGENVECTORL:
            B = C = Iv = MESS = MSS = 0; memset(eigen_work, 0, sizeof(double) * 4 * max_dim);
            eigen_decompose(N, N, &A.a[0][0], &MESS.a[0][0], &Iv.a[0][0], &C.a[0][0], &MSS.a[0][0], eigen_work, 4*max_dim);
            for (int i=0; i<C.n; i++){ double scaling = 0; for (int j=0; j<C.n; j++) scaling += C.a[i][j] * MSS.a[i][j];
                scaling = sqrt(fabs(scaling)); if (scaling!=0) for (int j=0; j<C.n; j++) C.a[i][j] /= scaling;
            }
            C.Transpose();
            eigen_sort(N, C, &MESS.a[0][0], MaxS, MinS);
            break;
        case CMD_EIGENVECTORR:
            B = C = Iv = MESS = MSS = 0; memset(eigen_work, 0, sizeof(double) * 4 * max_dim);
            eigen_decompose(N, N, &A.a[0][0], &MESS.a[0][0], &Iv.a[0][0], &MSS.a[0][0], &C.a[0][0], eigen_work, 4*max_dim);
            for (int i=0; i<C.n; i++){ double scaling = 0; for (int j=0; j<C.n; j++) scaling += C.a[i][j] * MSS.a[i][j];
                scaling = sqrt(fabs(scaling)); if (scaling!=0) for (int j=0; j<C.n; j++) C.a[i][j] /= scaling;
            }
            C.Transpose();
            eigen_sort(N, C, &MESS.a[0][0], MaxS, MinS);
            break;
        case CMD_EADD       : for (int y=0; y<N; y++) for (int x=0; x<N; x++) *C.e(y,x) = *A.e(y,x) + *B.e(y,x); break;
        case CMD_ESUB       : for (int y=0; y<N; y++) for (int x=0; x<N; x++) *C.e(y,x) = *A.e(y,x) - *B.e(y,x); break;
        case CMD_EPRODUCT   : for (int y=0; y<N; y++) for (int x=0; x<N; x++) *C.e(y,x) = *A.e(y,x) * *B.e(y,x); break;
        case CMD_EOVER      : for (int y=0; y<N; y++) for (int x=0; x<N; x++) *C.e(y,x) = *A.e(y,x) / *B.e(y,x); break;
        case CMD_F_ARRANGE  : rearrange(&B, &A, &C); break;
        case CMD_SUBMATRIX  : {
            int sub_dim = *B.e(0, 0);
            if (sub_dim>B.n)sub_dim=B.n; B = 0; for (int i=0;i<sub_dim*sub_dim; i++) B.m[i] = B.m[i] = floor(i/sub_dim)*B.n + i%sub_dim + 1;
            command = CMD_PRINT;
            break;
          }
        case CMD_DIAGONAL   : {
            B = 0; for (int i=0;i<B.n; i++) B.m[i] = i*B.n + i + 1;
            command = CMD_PRINT;
            break;
          }
        case CMD_INTEGRATE:
            MS += A; C = MS;
            break;
        case CMD_DIAG_TO_ROWS: for (int y=0; y<N; y++) for (int x=0; x<N; x++) *C.e(y,x) = *A.e(x,x); break;
        case CMD_DIAG_TO_COLS: for (int y=0; y<N; y++) for (int x=0; x<N; x++) *C.e(y,x) = *A.e(y,y); break;
        case CMD_ROW1_TO_ROWS: for (int y=0; y<N; y++) for (int x=0; x<N; x++) *C.e(y,x) = *A.e(0,x); break;
        case CMD_ROW1_TO_DIAG: C = 0; for (int x=0; x<N; x++) *C.e(x,x) = *A.e(0,x); break;
        case CMD_COL1_TO_COLS: for (int y=0; y<N; y++) for (int x=0; x<N; x++) *C.e(y,x) = *A.e(y,0); break;
        case CMD_COL1_TO_DIAG: C = 0; for (int y=0; y<N; y++) *C.e(y,y) = *A.e(y,0); break;
        }
        if (output_type==3){
        } else if (output_type==2){
            if (fail_inverse){
                printf(fmt_text, "nan");
            } else {
                printf(fmt_double, value);
            }
            printf("\n");
        } else if (output_type==4){     // show A with columns given in B
            for (int i=0; i<A.n; i++) for (int j=0; j<A.n; j++) {
                int im = ((int) *B.e(i, j)) - 1; if (im<0 || im>=A.n*A.n) continue;
                if (fail_inverse) printf(fmt_text, "nan");
                else printf(fmt_double, *A.e(im/A.n,im%A.n));
            }
            printf("\n");
        } else {
            if (wolfram_output) printf("{");
            for (int i=0; i<C.n; i++){
                if (wolfram_output) printf("{");
                for (int j=0; j<C.n; j++){
                    if (fail_inverse){
                        printf(fmt_text, "nan");
                    } else {
                        printf(fmt_double, *C.e(i,j));
                    }
                    if (j+1<C.n){
                        if (wolfram_output) printf(",");
                        else if (tab_output) printf(" ");
                    }
                }
                if (wolfram_output) printf(i+1<C.n?"},":"}");
                printf("%s", comma);
            }
            if (wolfram_output) printf("}");
            printf("\n");
        }
        /*if (!file[0] && !file[1]) success[0] = success[1] = false;
            for (int i=0; i<C.n; i++){
                for (int j=0; j<C.n; j++){
                    if (fail_inverse){
                        printf(fmt_text, "nan");
                    } else {
                        printf(fmt_double, C.a[i][j]);// *C.e(i,j));
                    }
                    printf(" ");
                }
                printf("%s", comma);
            }
            printf("\n");
        }*/
        if (!file[0] && !file[1]) success[0] = success[1] = false;
    }

    if (output_type == 3){
        switch (command){
        case CMD_S_SUM     : C = MS; break;
        case CMD_S_AVG     : {
            if (data_count==0) C = 0; else { C = MS; C /= data_count; };
          } break;
        case CMD_S_STDEV  : {
            for (int i=0; i<MS.n; i++) for (int j=0; j<MS.n; j++){
                *C.e(i,j) = data_count==0? 0 : sqrt(fabs((*MESS.e(i,j) - *MS.e(i,j)**MS.e(i,j)/data_count)/data_count));
            }
          } break;
        case CMD_S_MAX     : C = MaxS; break;
        case CMD_S_MIN     : C = MinS; break;
        }
        for (int i=0; i<C.n; i++){
            for (int j=0; j<C.n; j++){
                printf(fmt_double, *C.e(i,j)); printf(" ");
            }
            printf("%s", comma);
        }
        printf("\n");
    }

    free(eigen_work);
    dispose_all(&A, &B, &C, &Iv, &T, file[0], file[1]);
    dispose_all(&MS, &MSS, &MESS, &MaxS, &MinS, null, null);
    mcontainer.dispose();
    return 0;
}
