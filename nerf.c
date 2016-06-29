#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "util.h"
#include "matrix.h"
#include "stex.h"

#define BUFF_SIZE 1024

double mfuz;
double mfuzval;
double epsilon;
double qexp;
double qexpval;
double beta;
int objc;
int clustc;
int max_iter;
st_matrix memb;
st_matrix dmatrix;
st_matrix global_dmatrix;
st_matrix *membvec;
st_matrix dists;

void init_memb() {
    size_t i;
    size_t k;
    double sum;
    double val;
    for(i = 0; i < objc; ++i) {
        sum = 0.0;
        for(k = 0; k < clustc; ++k) {
            val = rand();
            sum += val;
            set(&memb, i, k, val);
        }
        for(k = 0; k < clustc; ++k) {
            set(&memb, i, k, get(&memb, i, k) / sum);
        }
    }
}

void hard_init_memb(int *class) {
    size_t i;
    size_t k;
    for(i = 0; i < objc; ++i) {
        for(k = 0; k < clustc; ++k) {
            set(&memb, i, k, 0.0);
        }
        set(&memb, i, class[i], 1.0);
    }
}

void print_memb(st_matrix *memb) {
	printf("Membership:\n");
	size_t i;
	size_t k;
	double sum;
    double val;
	for(i = 0; i < objc; ++i) {
		printf("%u: ", i);
		sum = 0.0;
		for(k = 0; k < clustc; ++k) {
            val = get(memb, i, k);
			printf("%lf ", val);
			sum += val;
		}
		printf("[%lf]", sum);
		if(!deq(sum, 1.0)) {
			printf("*\n");
		} else {
			printf("\n");
		}
	}
}

double adequacy() {
    size_t h;
    size_t i;
    size_t k;
    double sum_num;
    double sum_den;
    double adeq = 0.0;
    for(k = 0; k < clustc; ++k) {
        sum_num = 0.0;
        sum_den = 0.0;
        for(i = 0; i < objc; ++i) {
            for(h = 0; h < objc; ++h) {
                sum_num += pow(get(&memb, i, k), mfuz) *
                    pow(get(&memb, h, k), mfuz) * get(&dmatrix, i, h);
            }
            sum_den += pow(get(&memb, i, k), mfuz);
        }
        adeq += (sum_num / (2.0 * sum_den));
    }
    return adeq;
}

void global_dissim() {
    size_t h;
    size_t i;
    for(i = 0; i < objc; ++i) {
        for(h = 0; h < objc; ++h) {
            if(i != h) {
                set(&global_dmatrix, i, h,
                        get(&dmatrix, i, h) + beta);
            } else {
                set(&global_dmatrix, i, h, 0.0);
            }
        }
    }
//    printf("global_dmatrix:\n");
//    print_st_matrix(&global_dmatrix, 7, true);
}

void compute_membvec() {
    size_t k;
    size_t i;
    double sum_den;
    double val;
    for(k = 0; k < clustc; ++k) {
        sum_den = 0.0;
        for(i = 0; i < objc; ++i) {
            val = pow(get(&memb, i, k), mfuz);
            set(&membvec[k], i, 0, val);
            sum_den += val; 
        }
//        val = 0.0;
        for(i = 0; i < objc; ++i) {
            set(&membvec[k], i, 0, get(&membvec[k], i, 0) / sum_den);
//            val += get(&membvec[k], i, 0);
        }
//        printf("membvec[%d]:\n", k);
//        print_st_matrix(&membvec[k], 7, true);
//        printf("sum: %lf\n", val);
    }
}

bool compute_dists() {
    size_t i;
    size_t k;
    st_matrix membvec_trans;
    init_st_matrix(&membvec_trans, 1, objc);
    st_matrix aux_mtx;
    init_st_matrix(&aux_mtx, 1, objc);
    st_matrix aux_mtx2;
    init_st_matrix(&aux_mtx2, 1, 1);
    st_matrix term1;
    init_st_matrix(&term1, objc, 1);
    double term2;
    bool hasneg = false;
    for(k = 0; k < clustc; ++k) {
        transpose_(&membvec_trans, &membvec[k]);
        mtxmult_(&aux_mtx, &membvec_trans, &global_dmatrix);
        mtxmult_(&aux_mtx2, &aux_mtx, &membvec[k]);
        term2 = get(&aux_mtx2, 0, 0) * 0.5;
        mtxmult_(&term1, &global_dmatrix, &membvec[k]);
        for(i = 0; i < objc; ++i) {
            set(&dists, k, i, get(&term1, i, 0) - term2);
            if(!hasneg && get(&dists, k, i) < 0.0) {
                hasneg = true;
            }
        }
    }
    free_st_matrix(&membvec_trans);
    free_st_matrix(&aux_mtx);
    free_st_matrix(&aux_mtx2);
    free_st_matrix(&term1);
    return hasneg;
}

double compute_deltabeta() {
//    printf("Computing deltabeta...\n");
    size_t i;
    size_t k;
    double val;
    double deltabeta;
    bool first = true;
    double idcol[objc];
    for(i = 0; i < objc; ++i) {
        idcol[i] = 0.0;
    }
    for(i = 0; i < objc; ++i) {
        idcol[i] = 1.0;
        for(k = 0; k < clustc; ++k) {
//            val = -2.0 * (get(&dists, k, i) /
//                euclid_dist(membvec[k].mtx, idcol, objc));
            val = pow(euclid_dist(membvec[k].mtx, idcol, objc), 2.0);
//            val = minkowski(membvec[k].mtx, idcol, objc, 2.0);
            if(val != 0.0) {
                val = (-2.0 * get(&dists, k, i)) / val;
                if(first || val > deltabeta) {
                    deltabeta = val;
                    first = false;
                }
//                printf("Current: %lf\nBest: %lf\n", val, deltabeta);
            }
        }
        idcol[i] = 0.0;
    }
    return deltabeta;
}

bool adjust_dists() {
    printf("Adjusting distances...\n");
    int i;
    size_t k;
    double deltabeta = compute_deltabeta();
    printf("deltabeta: %.15lf\n", deltabeta);
    beta += deltabeta;
    printf("beta: %.15lf\n", beta);
    deltabeta /= 2.0;
    double idcol[objc];
    for(i = 0; i < objc; ++i) {
        idcol[i] = 0.0;
    }
    bool hasneg = false;
    for(i = 0; i < objc; ++i) {
        for(k = 0; k < clustc; ++k) {
            idcol[i] = 1.0;
//            printf("%lf + %lf * %lf\n", get(&dists, k, i), deltabeta,
//                    euclid_dist(membvec[k].mtx, idcol, objc));
            set(&dists, k, i, get(&dists, k, i) + deltabeta *
//                    minkowski(membvec[k].mtx, idcol, objc, 2.0));
                    pow(euclid_dist(membvec[k].mtx, idcol, objc), 2.0));
            idcol[i] = 0.0;
//            if(!hasneg && dlt(get(&dists, k, i), 0.0)) {
            if(!hasneg && get(&dists, k, i) < 0.0) {
                hasneg = true;
            }
        }
    }
    return hasneg;
}

void update_memb() {
    size_t c;
    size_t i;
    size_t k;
    double val;
    int zerovalc;
    for(i = 0; i < objc; ++i) {
        zerovalc = 0;
        for(k = 0; k < clustc; ++k) {
            if(!(get(&dists, k, i) > 0.0)) {
                ++zerovalc;
            }
        }
        if(zerovalc) {
            printf("Msg: there is at least one zero val for d[%d]."
                    "\n", i);
            val = 1.0 / ((double) zerovalc);
            for(k = 0; k < clustc; ++k) {
                if(get(&dists, k, i) > 0.0) {
                    set(&memb, i, k, 0.0);
                } else {
                    set(&memb, i, k, val);
                }
            }
        } else {
            for(k = 0; k < clustc; ++k) {
                val = 0.0;
                for(c = 0; c < clustc; ++c) {
//                    val += get(&dists, k, i) / get(&dists, c, i);
                    val += pow(get(&dists, k, i) / get(&dists, c, i),
                            mfuzval);
                }
//                set(&memb, i, k, 1.0 / pow(val, mfuzval));
                set(&memb, i, k, 1.0 / val);
            }
        }
    }
}

double run() {
//    int initvec[] = {0, 0, 1};
//    hard_init_memb(initvec);
    init_memb();
    print_memb(&memb);
    beta = 0.0;
    double adeq = adequacy();
    printf("Adequacy: %.15lf\n", adeq);
    double prev_iter_adeq;
    double adeq_diff;
    size_t iter = 1;
//    st_matrix prev_memb;
//    init_st_matrix(&prev_memb, objc, clustc);
    do {
        printf("Iteration %d:\n", iter);
        prev_iter_adeq = adeq;
        global_dissim();
        compute_membvec();
        if(compute_dists()) {
            do {
                printf("Distances:\n");
                print_st_matrix(&dists, 10, true);
            } while(adjust_dists());
        }
        printf("Distances:\n");
        print_st_matrix(&dists, 10, true);
//        mtxcpy(&prev_memb, &memb);
        update_memb();
        print_memb(&memb);
        adeq = adequacy();
        printf("Adequacy: %.15lf\n", adeq);
        adeq_diff = prev_iter_adeq - adeq;
//        printf("%.15lf %.15lf %.15lf\n", prev_iter_adeq, adeq, adeq_diff);
        if(adeq_diff < 0.0) {
            adeq_diff = fabs(adeq_diff);
            printf("Warn: current iteration adequacy is greater "
                    "than previous (%.15lf).\n", adeq_diff);
        }
        if(adeq_diff < epsilon) {
            printf("Adequacy difference threshold reached.\n");
            break;
        }
//        if(mtxeq(&prev_memb, &memb)) {
//            printf("Fuzzy memberships did not change.\n");
//            break;
//        }
        if(++iter > max_iter) {
            printf("Maximum number of iterations reached.\n");
            break;
        }
    } while(true);
//    free_st_matrix(&prev_memb);
    printf("Final beta: %.10lf\n", beta);
    return adeq;
}

int main(int argc, char **argv) {
    FILE *cfgfile = fopen(argv[1], "r");
    if(!cfgfile) {
        printf("Error: could not open config file.\n");
        return 1;
    }
    fscanf(cfgfile, "%d", &objc);
    if(objc <= 0) {
        printf("Error: objc <= 0.\n");
        fclose(cfgfile);
        return 1;
    }
    // reading labels
    int classc;
    int labels[objc];
    fscanf(cfgfile, "%d", &classc);
    size_t i;
    for(i = 0; i < objc; ++i) {
        fscanf(cfgfile, "%d", &labels[i]);
    }
    // reading labels end
    char filename[BUFF_SIZE];
    fscanf(cfgfile, "%s", filename);
    char outfilename[BUFF_SIZE];
    fscanf(cfgfile, "%s", outfilename);
    fscanf(cfgfile, "%d", &clustc);
    if(clustc <= 0) {
        printf("Error: clustc <= 0.\n");
        fclose(cfgfile);
        return 1;
    }
    int insts;
    fscanf(cfgfile, "%d", &insts);
    if(insts <= 0) {
        printf("Error: insts <= 0.\n");
        fclose(cfgfile);
        return 1;
    }
    fscanf(cfgfile, "%d", &max_iter);
    if(insts <= 0) {
        printf("Error: max_iter <= 0.\n");
        fclose(cfgfile);
        return 1;
    }
    fscanf(cfgfile, "%lf", &epsilon);
    if(dlt(epsilon, 0.0)) {
        printf("Error: epsilon < 0.\n");
        fclose(cfgfile);
        return 1;
    }
    fscanf(cfgfile, "%lf", &mfuz);
    if(dlt(mfuz, 1.0)) {
        printf("Error: mfuz < 1.0.\n");
        fclose(cfgfile);
        return 1;
    }
    fscanf(cfgfile, "%lf", &qexp);
    if(dlt(qexp, 1.0)) {
        printf("Error: qexp < 1.0.\n");
        fclose(cfgfile);
        return 1;
    }
    fclose(cfgfile);
    freopen(outfilename, "w", stdout);
    printf("###Configuration summary:###\n");
    printf("Number of objects: %d\n", objc);
    printf("Number of clusters: %d\n", clustc);
    printf("Number of instances: %d\n", insts);
    printf("Maximum interations: %d\n", max_iter);
    printf("Parameter m: %lf\n", mfuz);
    printf("Parameter q: %lf\n", qexp);
    printf("############################\n");
    st_matrix best_memb;
    st_matrix best_dists;
    // memory allocation start
    init_st_matrix(&dmatrix, objc, objc);
    init_st_matrix(&global_dmatrix, objc, objc);
    init_st_matrix(&memb, objc, clustc);
    init_st_matrix(&best_memb, objc, clustc);
    size_t k;
    membvec = malloc(sizeof(st_matrix) * clustc);
    for(k = 0; k < clustc; ++k) {
        init_st_matrix(&membvec[k], objc, 1);
    }
    init_st_matrix(&dists, clustc, objc);
    init_st_matrix(&best_dists, clustc, objc);
    // memory allocation end
    if(!load_data(filename, &dmatrix)) {
        printf("Error: could not load matrix file.\n");
        goto END;
    }
    mfuzval = 1.0 / (mfuz - 1.0);
    qexpval = 1.0 / (qexp - 1.0);
    srand(time(NULL));
    size_t best_inst;
    double best_inst_adeq;
    double best_beta;
    double cur_inst_adeq;
    for(i = 1; i <= insts; ++i) {
        printf("Instance %d:\n", i);
        cur_inst_adeq = run();
        if(i == 1 || cur_inst_adeq < best_inst_adeq) {
            mtxcpy(&best_memb, &memb);
            mtxcpy(&best_dists, &dists);
            best_inst_adeq = cur_inst_adeq;
            best_inst = i;
            best_beta = beta;
        }
    }
	printf("\n");
    printf("Best adequacy %.15lf on instance %d.\n", best_inst_adeq,
            best_inst);
    printf("\n");
    printf("beta: %.10lf\n", best_beta);
    print_memb(&best_memb);
    printf("\nPartition coefficient: %.10lf\n", partcoef(&best_memb));
    printf("Modified partition coefficient: %.10lf\n", modpcoef(&best_memb));
    printf("Partition entropy: %.10lf\n", partent(&best_memb));
    int *pred = defuz(&best_memb);
    st_matrix *best_dists_t = transpose(&best_dists);
    printf("\nDistances:\n");
    print_st_matrix(best_dists_t, 4, true);
    silhouet *sil = simplesil(pred, best_dists_t);
    printf("\nSimple silhouette:\n");
    print_silhouet(sil);
    free_silhouet(sil);
    free(sil);
    free_st_matrix(best_dists_t);
    free(best_dists_t);
    st_matrix *groups = asgroups(pred, objc, classc);
    printf("\nPartitions:\n");
    print_groups(groups);
    sil = crispsil(groups, &dmatrix);
    printf("\nCrisp silhouette:\n");
    print_silhouet(sil);
    silhouet *fsil = fuzzysil(sil, groups, &best_memb, 2.0);
    printf("\nFuzzy silhouette:\n");
    print_silhouet(fsil);
    free_silhouet(sil);
    free(sil);
    free_silhouet(fsil);
    free(fsil);
    free(pred);
    free_st_matrix(groups);
    free(groups);
END:
    fclose(stdout);
    free_st_matrix(&dmatrix);
    free_st_matrix(&global_dmatrix);
    free_st_matrix(&memb);
    free_st_matrix(&best_memb);
    for(k = 0; k < clustc; ++k) {
        free_st_matrix(&membvec[k]);
    }
    free(membvec);
    free_st_matrix(&dists);
    free_st_matrix(&best_dists);
    return 0;
}
