#include <bemtool/miscellaneous/htool_wrap.hpp>
#include <bemtool/tools.hpp>
#include <htool/clustering/clustering.hpp>
#include <htool/clustering/tree_builder/direction_computation.hpp>
#include <htool/clustering/tree_builder/splitting.hpp>
#include <htool/distributed_operator/distributed_operator.hpp>
#include <htool/solvers/ddm.hpp>
// #include <bemtool-tests/tools.hpp>

using namespace bemtool;
using namespace htool;

int main(int argc, char *argv[]) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the rank of the process
    int rank, sizeWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);

    // Command line
    // Check the number of parameters
    if (argc < 3) {
        // Tell the user how to run the program
        std::cerr << "Usage: " << argv[0] << " meshname outputpath" << std::endl;
        /* "Usage messages" are a conventional way of telling the user
         * how to run a program if they enter the command incorrectly.
         */
        return 1;
    }
    std::string meshname   = argv[1];
    std::string outputpath = argv[2];
    double kappa           = 10;
    int overlap            = 4;
    R3 dir;
    dir[0] = 1. / std::sqrt(2);
    dir[1] = 1. / std::sqrt(2);
    dir[2] = 0;

    // HTOOL
    double epsilon = 0.1;
    double eta     = -1;

    // HPDDM
    HPDDM::Option &opt = *HPDDM::Option::get();
    opt.parse(argc, argv);
    if (rank != 0)
        opt.remove("verbosity");

    // Mesh
    if (rank == 0)
        std::cout << "Loading mesh" << std::endl;
    Geometry node(meshname);
    Mesh1D mesh;
    mesh.Load(node, 0);
    Orienting(mesh);
    mesh                   = unbounded;
    int nb_elt             = NbElt(mesh);
    std::vector<R3> normal = NormalTo(mesh);

    // Dof
    if (rank == 0)
        std::cout << "Create dof" << std::endl;
    Dof<P1_1D> dof(mesh);
    int nb_dof = NbDof(dof);
    std::vector<double> x(3 * nb_dof);
    for (int i = 0; i < nb_dof; i++) {
        x[3 * i + 0] = dof(((dof.ToElt(i))[0])[0])[((dof.ToElt(i))[0])[1]][0];
        x[3 * i + 1] = dof(((dof.ToElt(i))[0])[0])[((dof.ToElt(i))[0])[1]][1];
        x[3 * i + 2] = dof(((dof.ToElt(i))[0])[0])[((dof.ToElt(i))[0])[1]][2];
    }
    WritePointValGmsh(dof, (outputpath + "/mesh.msh").c_str(), std::vector<int>(nb_dof, 1));

    if (rank == 0) {
        htool::vector_to_bytes(x, outputpath + "/geometry.bin");
    }

    // Clustering
    if (rank == 0)
        std::cout << "Creating cluster tree" << std::endl;
    ClusterTreeBuilder<double, ComputeLargestExtent<double>, GeometricSplitting<double>> target_recursive_build_strategy(nb_dof, 3, x.data(), 2, sizeWorld);
    std::shared_ptr<const Cluster<double>> target_root_cluster = std::make_shared<const Cluster<double>>(target_recursive_build_strategy.create_cluster_tree());

    // Generator
    if (rank == 0)
        std::cout << "Creating generators" << std::endl;
    BIO_Generator<HE_HS_2D_P1xP1, P1_1D> generator_W(target_root_cluster->get_permutation(), target_root_cluster->get_permutation(), dof, kappa);

    // Distributed operator
    if (rank == 0)
        std::cout << "Building Hmatrix" << std::endl;
    DistributedOperator<Cplx> W(target_root_cluster, target_root_cluster);
    W.build_default_hierarchical_approximation(generator_W, epsilon, eta);

    // Right-hand side
    if (rank == 0)
        std::cout << "Building rhs" << std::endl;
    std::vector<Cplx> rhs(nb_dof, 0);
    std::vector<double> rhs_real(nb_dof, 0), rhs_abs(nb_dof, 0);
    for (int i = 0; i < nb_elt; i++) {
        const N2 &jdof          = dof[i];
        const array<2, R3> xdof = dof(i);
        R2x2 M_local            = MassP1(mesh[i]);
        C2 Uinc;
        Uinc[0] = iu * kappa * (dir, normal[i]) * exp(iu * kappa * (xdof[0], dir));
        Uinc[1] = iu * kappa * (dir, normal[i]) * exp(iu * kappa * (xdof[1], dir));

        for (int k = 0; k < 2; k++) {
            rhs[jdof[k]] -= (M_local(k, 0) * Uinc[0] + M_local(k, 1) * Uinc[1]);
        }
    }

    if (rank == 0) {
        htool::vector_to_bytes(rhs, outputpath + "/rhs.bin");
    }

    // Overlap
    if (rank == 0)
        std::cout << "Building partitions" << std::endl;
    std::vector<int> cluster_to_ovr_subdomain;
    std::vector<int> ovr_subdomain_to_global;
    std::vector<int> neighbors;
    std::vector<std::vector<int>> intersections;

    Partition(target_recursive_build_strategy.get_partition(), target_root_cluster->get_permutation(), dof, cluster_to_ovr_subdomain, ovr_subdomain_to_global, neighbors, intersections);

    save_cluster_tree(*target_root_cluster, outputpath + "/cluster_" + htool::NbrToStr<int>(sizeWorld));
    htool::vector_to_bytes(cluster_to_ovr_subdomain, outputpath + "/cluster_to_ovr_subdomain_" + htool::NbrToStr(sizeWorld) + "_" + htool::NbrToStr(rank) + ".bin");
    htool::vector_to_bytes(ovr_subdomain_to_global, outputpath + "/ovr_subdomain_to_global_" + htool::NbrToStr(sizeWorld) + "_" + htool::NbrToStr(rank) + ".bin");
    htool::vector_to_bytes(neighbors, outputpath + "/neighbors_" + htool::NbrToStr(sizeWorld) + "_" + htool::NbrToStr(rank) + ".bin");

    for (int i = 0; i < intersections.size(); i++) {
        htool::vector_to_bytes(intersections[i], outputpath + "/intersections_" + htool::NbrToStr(sizeWorld) + "_" + htool::NbrToStr(rank) + "_" + htool::NbrToStr(i) + ".bin");
    }

    // Solve
    std::vector<Cplx> sol(nb_dof, 0);
    std::vector<double> sol_abs(nb_dof), sol_real(nb_dof);
    htool::DDM<Cplx> ddm(generator_W, &W, ovr_subdomain_to_global, cluster_to_ovr_subdomain, neighbors, intersections);
    ddm.facto_one_level();
    ddm.solve(rhs.data(), sol.data());
    ddm.print_infos();

    if (rank == 0) {
        htool::vector_to_bytes(sol, outputpath + "/sol.bin");
    }

    // Matrix
    std::vector<int> identity(nb_dof);
    std::iota(identity.begin(), identity.end(), int(0));
    htool::Matrix<Cplx> to_dense(nb_dof, nb_dof);
    generator_W.copy_submatrix(nb_dof, nb_dof, identity.data(), identity.data(), to_dense.data());
    to_dense.matrix_to_bytes("matrix.bin");

    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}
