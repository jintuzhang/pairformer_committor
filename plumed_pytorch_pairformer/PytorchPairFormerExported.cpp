/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Copyright (c) 2022-2026 of Luigi Bonati, Enrico Trizio and Jintu Zhang.

The pytorch module is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The pytorch module is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#include <algorithm>
#include <string>
#ifdef __PLUMED_HAS_LIBTORCH

#include <cmath>
#include <memory>
#include <fstream>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/jit_log.h>
#if __has_include("torch/csrc/inductor/aoti_package/model_package_loader.h")
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#else
#error Can not find header file: <torch/csrc/inductor/aoti_package/model_package_loader.h> ! Is your LibTorch too old?
#endif

#include "core/PlumedMain.h"
#include "colvar/Colvar.h"
#include "colvar/ActionRegister.h"
#include "tools/NeighborList.h"
#include "tools/Communicator.h"
#include "tools/OpenMP.h"
#include "tools/File.h"
#include "tools/PDB.h"

using namespace std;

namespace PLMD {

namespace colvar {

namespace pytorch_pairformer_exported {

//+PLUMEDOC PYTORCH_PAIRFORMER_EXPORTED PYTORCH_PAIRFORMER_EXPORTED
/*
Load a Pairformer model exported with the
`mlcolvar.pairformer.utils.export.export()` method.

This module uses a fixed length unit of _Angstrom_. Thus, the Pairformer model
read by this module should be trained under the same unit convention. Besides,
the module constructs node attributes w.r.t to the atomic types. As a result,
this module require a PDB file which records names of _ALL_ atoms in the system
(the STRUCTURE keyword). Note that the atom names in this PDB file could
_ONLY_ be element symbols, e.g.:
\auxfile{plumed_topo.pdb}
ATOM      1  H   ACE A   1      15.100  12.940  29.390  1.00  0.00           H
ATOM      2  C   ACE A   1      14.970  13.860  29.960  1.00  0.00           C
ATOM      3  H   ACE A   1      15.720  13.820  30.760  1.00  0.00           H
ATOM      4  H   ACE A   1      13.980  13.920  30.410  1.00  0.00           H
ATOM      5  C   ACE A   1      15.300  15.070  29.100  1.00  0.00           C
\endauxfile

The module characterize geometric of the system by fully connected distance
matrix formed by atoms inside the selected atom group, as well the corresponded
residue/atom types. Such an atom group is defined by the GROUPA keyword.
To include solvent effects in the ML model, this module also supports using
coordination numbers as descriptors. Such coordination numbers are defined
between solvent atoms (the GROUPB keyword) and geometric centers of center
groups (one or multiple GROUPC keywords). When coordination number descriptors
are required, a neighbor list is used to accelerate the computation.

This module also support committor calculations. When the input PyTorch model
is a committor model, the outputs will assign the zeta value to the first
output (node-0) and the q value to the second output (node-1). If the
`calculate_k_bias` option (along with other Kolmogorov bias calculation
parameters) is given when exporting, the module will also alculate the committor
bias and assign it with a label of kbias. These information will be shown in the
log as well.

Besides, dtype and running device of the model will be fixed after export, which
means that one can not exporting a model stored on CPU and then inference on GPU.
What's more, The exported models are generally MUCH FASTER when running on GPUs.
See the docstring of the `mlcolvar.pairformer.utils.export.export()` method for
details.

Note that this function requires \ref installation-libtorch LibTorch C++ library.
Check the instructions in the \ref PYTORCH page to enable the module.
Specifically, we encourage the user to install the GPU-enabled version of
LibTorch, when dealing with large input structures.

\par Examples
The following example instructs plumed to evaluate the Pairformer model using the atoms 1-10.
\plumedfile
PYTORCH_PAIRFORMER_EXPORTED ...
  GROUPA=1-10
  MODEL=model.pt2
  STRUCTURE=plumed_topo.pdb
  LABEL=pf
... PYTORCH_PAIRFORMER_EXPORTED
\endplumedfile

The following example instructs plumed to do the same calculation as the above example, and will add an OPES bias potential on the CV.
\plumedfile
PYTORCH_PAIRFORMER_EXPORTED ...
  GROUPA=1-10
  MODEL=model.pt2
  STRUCTURE=plumed_topo.pdb
  LABEL=pf
... PYTORCH_PAIRFORMER_EXPORTED

OPES_METAD ...
  LABEL=opes
  ARG=pf.node-0
  FILE=KERNELS
  PACE=500
  TEMP=300
  BARRIER=35
... OPES_METAD
\endplumedfile

The following example instructs plumed to do the same calculation as the above example, but will sample Kolmogorov's transition state ensemble.
\plumedfile
PYTORCH_PAIRFORMER_EXPORTED ...
  GROUPA=1-10
  MODEL=model.pt2
  STRUCTURE=plumed_topo.pdb
  LABEL=pf
... PYTORCH_PAIRFORMER_EXPORTED

BIASVALUE ARG=pf.kbias LABEL=vk
\endplumedfile

\par Examples
The following example instructs plumed to evaluate the Pairformer model using the atoms 1-10, and uss coordination numbers between atoms 100 to 200 and centers of atoms 2-4/6-8 as extra descriptors.
\plumedfile
PYTORCH_PAIRFORMER_EXPORTED ...
  GROUPA=1-10
  GROUPB=100-200
  GROUPC0=2-4
  GROUPC1=6-8
  MODEL=model.pt2
  STRUCTURE=plumed_topo.pdb
  LABEL=pf
... PYTORCH_PAIRFORMER_EXPORTED
\endplumedfile

*/
//+ENDPLUMEDOC


class PytorchPairFormerExported: public Colvar
{
  int n_cvs = 0;
  int n_centers = 0;
  int n_embeddings = 0;
  int n_atoms_padded = 0;
  int n_atoms_padded_environment = 0;
  int n_residues_padded = 0;
  bool pbc = true;
  bool serial = false;
  bool firsttime = true;
  bool invalidate_list = true;
  bool is_committor = false;
  bool k_bias = false;
  bool kb_weighted = false;
  bool kb_truncated = false;
  bool kb_truncated_cos = false;
  bool ignore_excessed_nl_atoms = false;
  double cutoff = 0.0; // In PLUMED length unit
  double kb_lambda = -1.0;
  double kb_epsilon = -1.0;
  double gradient_threshold = 1E20;
  std::string model_file_name;
  std::string structure_file_name;
  std::vector<int> system_node_types;
  std::vector<int> system_residue_types;
  std::vector<std::string> model_atom_names;
  std::vector<std::string> model_residue_names;
  std::vector<double> model_atomic_masses;
  std::vector<AtomNumber> atom_list_a;
  std::vector<AtomNumber> atom_list_b;
  std::vector<AtomNumber> atom_list_c;
  std::vector<std::vector<AtomNumber>> atom_lists_c;
  std::vector<int> atom_list_active; // local_ids
  std::unique_ptr<NeighborList> neighbor_list;
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> model;
  torch::ScalarType torch_float_dtype = torch::kFloat32;
  torch::Device device = c10::Device(torch::kCPU);
  torch::Tensor centers;
  torch::Tensor system_masks;
  torch::Tensor pair_masks;
  torch::Tensor residue_adjustency;
  const std::array<std::string, 3> implemented_embeddings = {
    "atom_names", "residue_names", "",
  };
  void check_embedding_is_implemented(std::string name);
  bool groups_have_intersection(void);
  bool group_c_is_in_group_a(void);
  void find_active_atoms(int n_threads);
  void clip_gradients(torch::Tensor &gradients);
  std::string trim(const std::string& str);

public:
  explicit PytorchPairFormerExported(const ActionOptions&);
  ~PytorchPairFormerExported();
  static void registerKeywords(Keywords& keys);
  void calculate() override;
  void prepare() override;
}; // class PytorchPairFormerExported

PLUMED_REGISTER_ACTION(PytorchPairFormerExported, "PYTORCH_PAIRFORMER_EXPORTED")

void PytorchPairFormerExported::registerKeywords(Keywords& keys)
{
  Colvar::registerKeywords(keys);

  keys.add(
    "atoms",
    "GROUPA",
    "First list of atoms (corresponding to the `system_selection` in mlcolvar)"
  );

  keys.add(
    "atoms",
    "GROUPB",
    "Second list of atoms (corresponding to the `environment_selection` in mlcolvar)"
  );

  keys.add(
    "atoms",
    "GROUPC",
    "Center atoms (corresponding to the `center_selections` in mlcolvar`)"
  );

  keys.add(
    "compulsory",
    "MODEL",
    "Filename of the PyTorch compiled model"
  );

  keys.add(
    "compulsory",
    "STRUCTURE",
    "PDB file name that contains the whole simulated system, with currect atom names and orders"
  );

  keys.add(
    "optional",
    "NL_STRIDE",
    "The frequency with which we are updating the atoms in the neighbor list"
  );

  keys.add(
    "optional",
    "GRADIENT_THRESHOLD",
    "Threshold of CV gradients"
  );

  keys.addFlag(
    "SERIAL",
    false,
    "Perform the calculation in serial - for debug purpose"
  );

  keys.addFlag(
    "IGNORE_EXCESSED_NL_ATOMS",
    false,
    "Ignore GROUPB atoms that have an index larger than `n_atoms_padded_environment`"
  );

  keys.addOutputComponent(
    "node",
    "default",
    "Model outputs"
  );

  keys.addOutputComponent(
    "kbias",
    "KBIAS",
    "Kolmogorov's bias potential $V_K$"
  );
}

PytorchPairFormerExported::PytorchPairFormerExported(const ActionOptions& ao):
  PLUMED_COLVAR_INIT(ao)
{
  // print libtorch version
  std::stringstream ss;
  ss << TORCH_VERSION_MAJOR << "." \
     << TORCH_VERSION_MINOR << "." \
     << TORCH_VERSION_PATCH;
  std::string version;
  ss >> version; // extract into the string.
  std::string version_info = "  LibTorch version: " + version + "\n";

  // parse input
  parseAtomList("GROUPA", atom_list_a);
  parseAtomList("GROUPB", atom_list_b);

  for(int i = 0; ; i++) {
    std::vector<AtomNumber> group;
    parseAtomList("GROUPC", i, group);
    if (group.empty())
      break;
    atom_lists_c.push_back(group);
  }

  parse("MODEL", model_file_name);

  parse("STRUCTURE", structure_file_name);

  int neighbor_list_stride = 1;
  parse("NL_STRIDE", neighbor_list_stride);
  if (neighbor_list_stride <= 0)
    plumed_merror("NL_STRIDE should be positive!");

  parse("GRADIENT_THRESHOLD", gradient_threshold);

  parseFlag("SERIAL", serial);

  bool nopbc = !pbc;
  parseFlag("NOPBC", nopbc);
  pbc = !nopbc;

  parseFlag("IGNORE_EXCESSED_NL_ATOMS", ignore_excessed_nl_atoms);

  checkRead();

  // check groups
  if (atom_list_b.size() > 0) {
    if (groups_have_intersection())
      plumed_merror("GROUPA can not intersect with GROUPB!");
    if (!group_c_is_in_group_a())
      plumed_merror("Not all atoms in GROUPC present in GROUPA!");
  } else {
    find_active_atoms(1);
  }

  // build center vatoms
  if (atom_list_b.size() > 0) {
    log << "  adding center vatoms ..." << "\n";
    log.setLinePrefix("PLUMED:     ");
    std::vector<std::string> center_names;
    for (int i = 0; i < (int)atom_lists_c.size(); i++) {
      std::string line = "CENTER ATOMS=";
      std::string center_name = getLabel() + "_cntr_" + std::to_string(i);
      for (auto j: atom_lists_c[i]) {
        line += std::to_string(j.serial()) + ",";
      }
      line = line + " LABEL=" + center_name;
      plumed.readInputLine(line);
      center_names.push_back(center_name);
    }
    interpretAtomList(center_names, atom_list_c);
    log.setLinePrefix("PLUMED: ");
    log << "  added center vatoms: ";
    for (size_t i = 0; i < atom_list_c.size(); i++)
      log << atom_list_c[i].serial() << " ";
    log << "\n";
  }

  // check structure file
  PDB pdb;
  FILE *fp = fopen(structure_file_name.c_str(), "r");
  if (fp != NULL) {
    pdb.readFromFilepointer(
      fp,
      atoms.usingNaturalUnits(),         // TODO: remove the `atoms.` prefix when release
      0.1 / atoms.getUnits().getLength() // TODO: remove the `atoms.` prefix when release
    );
    fclose(fp);
  } else {
    plumed_merror("Can not open PDB file: '" + structure_file_name + "'");
  }

  // deserialize the model from file
  try {
    model = Tools::make_unique<torch::inductor::AOTIModelPackageLoader>(
      model_file_name
    );
  } catch (const c10::Error& e) {
    plumed_merror(
      "Cannot load exported model file: '" + model_file_name + "'. Reason: " + e.what()
    );
  }

  // read information from the model
  auto metadata = model->get_metadata();

  // dtype/device
  bool use_cuda = false;
  std::string float_dtype_exported(metadata.at("float_dtype").c_str());
  if (float_dtype_exported == "32")
    torch_float_dtype = torch::kFloat32;
  else if (float_dtype_exported == "64")
    torch_float_dtype = torch::kFloat64;
  else
    plumed_merror("Unknown float dtype \"" + float_dtype_exported + "\" found in the exported model \"" + model_file_name + "\"!");
  std::string device_exported(metadata.at("AOTI_DEVICE_KEY").c_str());
  if (device_exported == "cuda") {
    if (!torch::cuda::is_available())
      plumed_merror("Exported model \"" + model_file_name + "\" requires running on CUDA, however CUDA device not found/LibTorch does not support CUDA!");
    device = c10::Device(torch::kCUDA);
    use_cuda = true;
  } else {
    device = c10::Device(torch::kCPU);
    use_cuda = false;
  }

  // CV size/position tensor size
  n_cvs = std::atoi(metadata.at("n_cvs").c_str());
  n_atoms_padded = std::atoi(
    metadata.at("n_atoms_padded").c_str()
  );
  n_atoms_padded_environment = std::atoi(
    metadata.at("n_atoms_padded_environment").c_str()
  );
  if (metadata.count("n_residues_padded"))
    n_residues_padded = std::atoi(
      metadata.at("n_residues_padded").c_str()
    );
  else
    n_residues_padded = 0;

  // embedding tables
  int n_atom_names = 0;
  int n_residue_names = 0;
  n_embeddings = std::atoi(metadata.at("n_embeddings").c_str());

  for (int64_t i = 0; i < n_embeddings; i++) {
    std::string embedding_name(
      metadata.at("embedding_" + to_string(i)).c_str()
    );
    check_embedding_is_implemented(embedding_name);

    if (embedding_name == "atom_names") {
      n_atom_names = std::atoi(metadata.at("n_atom_names").c_str());
      for (int64_t j = 0; j < n_atom_names; j++)
        model_atom_names.push_back(
          metadata.at("atom_names_" + to_string(j))
        );
    }
    if (embedding_name == "residue_names") {
      n_residue_names = std::atoi(metadata.at("n_residue_names").c_str());
      for (int64_t j = 0; j < n_residue_names; j++)
        model_residue_names.push_back(
          metadata.at("residue_names_" + to_string(j))
        );
    }
  }

  // static masks
  if (n_residues_padded > 0) {
    pair_masks = torch::zeros(
      {n_residues_padded, n_residues_padded}, torch::dtype(torch::kInt64)
    );
    system_masks = torch::zeros(
      {n_atoms_padded, 1}, torch::dtype(torch::kBool)
    );
  } else {
    pair_masks = torch::zeros(
      {n_atoms_padded, n_atoms_padded}, torch::dtype(torch::kInt64)
    );
    system_masks = torch::tensor(
      0, torch::dtype(torch::kBool)
    );
  }

  if (n_residues_padded > 0) {
    int count = 0;
    unsigned resseq = pdb.getResidueNumber(atom_list_a[0]);
    std::string chain_id = pdb.getChainID(atom_list_a[0]);
    residue_adjustency = torch::zeros(
      {n_residues_padded, n_atoms_padded}, torch::dtype(torch::kInt64)
    );
    for (size_t i = 0; i < atom_list_a.size(); i++) {
      unsigned resseq_i = pdb.getResidueNumber(atom_list_a[i]);
      std::string chain_id_i = pdb.getChainID(atom_list_a[i]);
      if ((resseq_i != resseq) || (chain_id_i != chain_id)) {
        resseq = resseq_i;
        chain_id = chain_id_i;
        count++;
        if (count >= n_residues_padded)
          plumed_merror("Number of residues " + std::to_string(count) + " is larger than the padding size " + std::to_string(n_residues_padded));
      }
      residue_adjustency[count][i] = 1;
    }
    pair_masks.index({
      torch::indexing::Slice(0, count + 1),
      torch::indexing::Slice(0, count + 1),
    }) = 1;
    system_masks.index({
      torch::indexing::Slice(0, atom_list_a.size()),
      torch::indexing::Slice(0, 1),
    }) = true;
  } else {
    residue_adjustency = torch::tensor(0, torch::dtype(torch::kInt64));
    pair_masks.index({
      torch::indexing::Slice(0, atom_list_a.size()),
      torch::indexing::Slice(0, atom_list_a.size()),
    }) = 1;
  }

  residue_adjustency = residue_adjustency.to(device);
  system_masks = system_masks.to(device);
  pair_masks = pair_masks.to(device);

  // check model type
  is_committor = metadata.at("is_committor") == "True";
  if (is_committor) {
    if (n_cvs != 2)
      plumed_merror("The committor model should output two values!");
    for (int64_t i = 0; i < n_atom_names; i++)
      model_atomic_masses.push_back(
        std::atof(metadata.at("atomic_masses_" + to_string(i)).c_str())
      );
    k_bias = metadata.at("calculate_k_bias") == "True";
    if (k_bias) {
      kb_lambda = std::atof(metadata.at("kb_lambda").c_str());
      kb_epsilon = std::atof(metadata.at("kb_epsilon").c_str());
      kb_weighted = metadata.at("kb_weighted") == "True";
      kb_truncated = metadata.at("kb_truncated") == "True";
      kb_truncated_cos = metadata.at("kb_truncated_cos") == "True";
    }
  }

  // summary/number of parameters/training time
  std::string model_summary(metadata.at("model_summary").c_str());
  std::string model_n_parameters(metadata.at("n_parameters").c_str());
  std::string model_training_time(metadata.at("training_time").c_str());
  std::string model_exporting_time(metadata.at("exporting_time").c_str());

  // check if we have gradients
  if (metadata.at("calculate_gradients") != "True")
      plumed_merror(
        "Exported model \"" + model_file_name + "\" does not contain gradients!"
      );

  // create system atomic numbers
  std::vector<int> atom_is_required(pdb.getAtomNumbers().size());
  for (size_t i = 0; i < atom_list_a.size(); i++) {
    int index = atom_list_a[i].index();
    atom_is_required[index] = 1;
  }

  if (n_atom_names > 0) {
    for (size_t i = 0; i < pdb.getAtomNumbers().size(); i++) {
      AtomNumber index = pdb.getAtomNumbers()[i];
      std::string name = pdb.getAtomName(index);
      auto iter = std::find(
        model_atom_names.begin(), model_atom_names.end(), name
      );
      if (iter == model_atom_names.end()) {
        if (atom_is_required[i])
          plumed_merror(
            "Atom '" + name + "' does not present in model " + model_file_name
          );
        else
          system_node_types.push_back(-1);
      } else {
        int node_type = std::distance(model_atom_names.begin(), iter);
        system_node_types.push_back(node_type);
      }
    }
  }

  // create residue name embedding
  if (n_residue_names > 0) {
    for (size_t i = 0; i < pdb.getAtomNumbers().size(); i++) {
      AtomNumber index = pdb.getAtomNumbers()[i];
      std::string name = trim(pdb.getResidueName(index));
      auto iter = std::find(
        model_residue_names.begin(), model_residue_names.end(), name
      );
      if (iter == model_residue_names.end()) {
        if (atom_is_required[i])
          plumed_merror(
            "Residue '" + name + "' does not present in model " + model_file_name
          );
        else
          system_residue_types.push_back(-1);
      } else {
        int residue_type = std::distance(model_residue_names.begin(), iter);
        system_residue_types.push_back(residue_type);
      }
    }
  }

  // centers
  n_centers = std::atoi(metadata.at("n_centers").c_str());
  if (n_centers == 0) {
    if (atom_list_b.size() > 0)
      plumed_merror(
        "Exported model \"" + model_file_name + "\" does not contain a CN model! However the `GROUPB` keyword is defined!"
      );
    if (atom_list_c.size() > 0)
      plumed_merror(
        "Exported model \"" + model_file_name + "\" does not contain a CN model! However the `CENTERS` keyword is defined!"
      );

    centers = torch::zeros({1}, torch::dtype(torch::kInt64));
    centers = centers.to(device);
  } else {
    if (atom_list_b.size() == 0)
      plumed_merror(
        "Exported model \"" + model_file_name + "\" contains a CN model! However the `GROUPB` keyword is not defined!"
      );
    if (atom_list_c.size() == 0)
      plumed_merror(
        "Exported model \"" + model_file_name + "\" contain a CN model! However the `CENTERS` keyword is not defined!"
      );
    else if ((int)atom_list_c.size() != n_centers)
      plumed_merror(
        "Exported model \"" + model_file_name + "\" requires "
        + std::to_string(n_centers) +
        " centers! However "
        + std::to_string(atom_list_c.size()) +
        " centers are given!"
      );

    cutoff = std::atof(metadata.at("d_max").c_str());
    cutoff = cutoff / atoms.getUnits().getLength() * 0.1; // TODO: remove the `atoms.` prefix when release
    neighbor_list = Tools::make_unique<NeighborList>(
      atom_list_c,
      atom_list_b,
      serial,
      false,
      pbc,
      getPbc(),
      comm,
      cutoff,
      neighbor_list_stride
    );
  }

  // create components
  if (!is_committor) {
    for (int i = 0; i < n_cvs; i++) {
      string name_comp = "node-" + std::to_string(i);
      addComponentWithDerivatives(name_comp);
      componentIsNotPeriodic(name_comp);
    }
  } else {
    string name_comp_z = "node-0";
    addComponentWithDerivatives(name_comp_z);
    componentIsNotPeriodic(name_comp_z);
    string name_comp_q = "node-1";
    addComponent(name_comp_q);
    componentIsNotPeriodic(name_comp_q);
    if (k_bias) {
      string name_comp_b = "kbias";
      addComponentWithDerivatives(name_comp_b);
      componentIsNotPeriodic(name_comp_b);
    }
  }

  // take atoms from plumed
  if (n_centers == 0)
    requestAtoms(atom_list_a);

  // make centers
  if (n_centers > 0) {
    int n_center_atoms_padded = 0;
    for (unsigned int j = 0; j < atom_lists_c.size(); j++) {
      if ((int)atom_lists_c[j].size() > n_center_atoms_padded)
        n_center_atoms_padded = atom_lists_c[j].size();
    }
    centers = -torch::ones(
      {n_centers, n_center_atoms_padded}, torch::dtype(torch::kInt64)
    );
    for (unsigned int j = 0; j < atom_lists_c.size(); j++) {
      for (unsigned int i = 0; i < atom_lists_c[j].size(); i++) {
        // NOTE: find index of center atoms from the system atom list
        auto iter = std::find(
          atom_list_a.begin(), atom_list_a.end(), atom_lists_c[j][i]
        );
        centers[j][i] = std::distance(atom_list_a.begin(), iter);
      }
    }
    centers = centers.unsqueeze(0);
    centers = centers.to(device);
  }

  // print log
  log.printf(version_info.data());
  log.printf("  Interface build time: %s %s\n", __DATE__, __TIME__);
  std::string thename = getLabel();
  log.printf(
    "  Will build inputs for PairFormer using %u system atoms\n",
    static_cast<unsigned>(atom_list_a.size())
  );
  log.printf("  Padded atom list size: %d\n", n_atoms_padded);
  log.printf("  System atom list (GROUPA):\n   ");
  for (unsigned int i = 0; i < atom_list_a.size(); i++) {
    if (((i + 1) % 10) == 0)
      log.printf("\n   ");
    log.printf(" %d", atom_list_a[i].serial());
  }
  log.printf("\n");
  if (n_centers > 0) {
    log.printf(
      "  Will build inputs for CN model using %u center and %u environment atoms\n",
      static_cast<unsigned>(atom_list_c.size()),
      static_cast<unsigned>(atom_list_b.size())
    );
    log.printf("  Center atom list:\n   ");
    for (unsigned int i = 0; i < atom_list_c.size(); i++) {
      if (((i + 1) % 10) == 0)
        log.printf("\n   ");
      log.printf(" %d", atom_list_c[i].serial());
    }
    log.printf("\n");
    log.printf("  Padded environment list size: %d\n", n_atoms_padded_environment);
    log.printf("  Environment atom list (GROUPB):\n   ");
    for (unsigned int i = 0; i < atom_list_b.size(); i++) {
      if (((i + 1) % 10) == 0)
        log.printf("\n   ");
      log.printf(" %d", atom_list_b[i].serial());
    }
    log.printf("\n");
    log.printf("  Neighbor List update stride: %d\n", neighbor_list_stride);
    log.printf("  CN model cutoff radius: %f (PLUMED length unit)\n", cutoff);
  }
  log.printf("  Is this a residue-level model: ");
  if (n_residues_padded > 0) {
    log.printf("yes\n");
    log.printf("  Padded token list size: %d\n", n_residues_padded);
    log.printf("  Residue mapping:\n");
    for (int i = 0; i < n_residues_padded; i++) {
      log.printf("    %d: ", i);
      if (residue_adjustency[i].sum().item<int64_t>() > 0) {
        for (int j = 0; j < n_atoms_padded; j++) {
          if (residue_adjustency[i][j].item<int64_t>() != 0)
            log.printf("%d ", atom_list_a[j].serial());
        }
      } else {
        log.printf("(null)");
      }
      log.printf("\n");
    }
  } else {
    log.printf("no\n");
  }
  log.printf("  Model atom names:\n    ");
  for (unsigned int i = 0; i < model_atom_names.size(); i++) {
    if (((i + 1) % 10) == 0)
      log.printf("\n    ");
    log.printf("%-5s", model_atom_names[i].c_str());
  }
  log.printf("\n");
  if (model_residue_names.size() > 0) {
    log.printf("  Model residue names:\n    ");
    for (unsigned int i = 0; i < model_residue_names.size(); i++) {
      if (((i + 1) % 10) == 0)
        log.printf("\n    ");
      log.printf("%-5s", model_residue_names[i].c_str());
    }
    log.printf("\n");
  }
  log.printf("  Boundary conditions: ");
  if (pbc)
    log.printf("periodic\n");
  else
    log.printf("non-periodic\n");
  log.printf("  Number of outputs: %d \n", n_cvs);
  log.printf("  Is this a committor model: ");
  if (is_committor)
    log.printf("yes\n");
  else
    log.printf("no\n");
  if (is_committor) {
    log.printf("  If sample Kolmogorov's ensemble: ");
    if (k_bias)
      log.printf("yes\n");
    else
      log.printf("no\n");
    if (k_bias) {
      log.printf("  If calculate truncated V_K: ");
        if (kb_truncated)
          log.printf("yes\n");
        else
          log.printf("no\n");
      log.printf("  If calculate truncated V_K using cosine function: ");
        if (kb_truncated_cos)
          log.printf("yes\n");
        else
          log.printf("no\n");
      log.printf("  If calculate mass-weighted V_K: ");
      if (kb_weighted) {
        log.printf("yes\n");
        log.printf("  Model atomic masses:\n   ");
        for (unsigned int i = 0; i < model_atomic_masses.size(); i++) {
          if (((i + 1) % 10) == 0)
            log.printf("\n   ");
          log.printf(" %.4f", model_atomic_masses[i]);
        }
        log.printf("\n");
      } else {
        log.printf("no\n");
      }
      log.printf("  LAMBDA    value for calculating V_K: %f\n", kb_lambda);
      log.printf("  EPSILON   value for calculating V_K: %e\n", kb_epsilon);
    }
    if (k_bias) {
      log << "  Output alignment: " + thename + ".kbias  -> V_K\n";
      log << "  Output alignment: " + thename + ".node-0 -> zeta\n";
    } else {
      log << "  Output alignment: " + thename + ".node-0 -> zeta\n";
    }
    log << "  Output alignment: " + thename + ".node-1 -> q (no grad)\n";
  }
  log.printf("  CV gradient threshold: %e\n", gradient_threshold);
  log.printf("  Will run on device: ");
  if (use_cuda)
    log.printf("CUDA\n");
  else
    log.printf("CPU (as required)\n");
  log << "  Model file name: " + model_file_name + "\n";
  log << "  Model training  time: " + model_training_time + "\n";
  log << "  Model exporting time: " + model_exporting_time + "\n";
  log << "  Model parameters: " + model_n_parameters + "\n";
  log << "  Model architecture: \n";
  log << model_summary;
  log << "  Bibliography: ";
  log << plumed.cite("Bonati, Trizio, Rizzi and Parrinello, J. Chem. Phys. 159, 014801 (2023)");
  log << plumed.cite("Bonati, Rizzi and Parrinello, J. Phys. Chem. Lett. 11, 2998-3004 (2020)");
  log << plumed.cite("Zhang et al., arXiv preprint arXiv:2606.31832 (2026)");
  log.printf("\n");
}

PytorchPairFormerExported::~PytorchPairFormerExported()
{
  return;
}

void PytorchPairFormerExported::prepare()
{
  if (n_centers > 0) {
    std::vector<AtomNumber> cn_atoms;

    if (neighbor_list->getStride() > 0) {
      if (firsttime || ((getStep() % neighbor_list->getStride()) == 0)) {
        cn_atoms = neighbor_list->getFullAtomList();
        invalidate_list = true;
        firsttime = false;
      } else {
        cn_atoms = neighbor_list->getReducedAtomList();
        invalidate_list = false;
        if (getExchangeStep())
          plumed_merror(
            "Neighbor lists should be updated on exchange steps - choose a NL_STRIDE which divides the exchange stride!"
          );
      }

      std::vector<AtomNumber> merge;
      merge.reserve(atom_list_a.size() + cn_atoms.size());
      merge.insert(merge.end(), atom_list_a.begin(), atom_list_a.end());
      merge.insert(merge.end(), cn_atoms.begin(), cn_atoms.end());
      requestAtoms(merge);

      if (getExchangeStep())
        firsttime = true;
    }
  }
}


void PytorchPairFormerExported::calculate()
{
  // get some common data
  int n_atoms = getNumberOfAtoms();
  std::vector<PLMD::Vector> x_local = getPositions();

  // threads
  int n_threads = OpenMP::getNumThreads();
  if (!serial)
    n_threads = std::min(n_threads, n_atoms);
  else
    n_threads = 1;

  // perform the size check
  if (system_node_types.size() != (size_t)atoms.getNatoms())
    plumed_merror(
      "Structure file '" +
      structure_file_name +
      "' has different number of atoms with the simulated system!"
    );

  int n_atoms_a = (int)atom_list_a.size();

  // update the neighbor list and number of atoms
  if (n_centers > 0) {
    if (neighbor_list->getStride() > 0 && invalidate_list) {
      std::vector<PLMD::Vector> x_local_environment(
        x_local.begin() + n_atoms_a, x_local.end()
      );
      neighbor_list->update(x_local_environment);
    }
    find_active_atoms(n_threads);
  }
  n_atoms = (int)atom_list_active.size();
  n_threads = std::min(n_threads, n_atoms);

  int n_neighbors = n_atoms - (int)atom_list_a.size();

  // get the unit
  double to_ang = 10 * atoms.getUnits().getLength(); // TODO: remove the `atoms.` prefix when release

  // get the positions
  // TODO: now, the positions used by the model file is in unit of Angstrom.
  // We should warn the users about this default
  torch::Tensor positions;
  std::vector<float> positions_vector_s(n_atoms_padded * 3);
  std::fill(positions_vector_s.begin(), positions_vector_s.end(), 0.0);
  #pragma omp parallel for num_threads(n_threads)
  for (int i = 0; i < n_atoms_a; i++) {
    int index = atom_list_active[i];
    positions_vector_s[i * 3 + 0] = x_local[index][0] * to_ang;
    positions_vector_s[i * 3 + 1] = x_local[index][1] * to_ang;
    positions_vector_s[i * 3 + 2] = x_local[index][2] * to_ang;
  }
  torch::Tensor positions_s = torch::from_blob(
    positions_vector_s.data(),
    n_atoms_padded * 3,
    torch::TensorOptions().dtype(torch::kFloat32)
  );
  positions_s = positions_s.to(device).to(torch_float_dtype);
  positions_s = positions_s.reshape({n_atoms_padded, 3});

  if (n_centers > 0) {
    std::vector<float> positions_vector_e(n_atoms_padded_environment * 3);
    std::fill(positions_vector_e.begin(), positions_vector_e.end(), 0.0);
    #pragma omp parallel for num_threads(n_threads)
    for (int i = n_atoms_a; i < n_atoms; i++) {
      int index = atom_list_active[i];
      positions_vector_e[(i - n_atoms_a) * 3 + 0] = x_local[index][0] * to_ang;
      positions_vector_e[(i - n_atoms_a) * 3 + 1] = x_local[index][1] * to_ang;
      positions_vector_e[(i - n_atoms_a) * 3 + 2] = x_local[index][2] * to_ang;
    }
    torch::Tensor positions_e = torch::from_blob(
      positions_vector_e.data(),
      n_atoms_padded_environment * 3,
      torch::TensorOptions().dtype(torch::kFloat32)
    );
    positions_e = positions_e.to(device).to(torch_float_dtype);
    positions_e = positions_e.reshape({n_atoms_padded_environment, 3});
    positions = torch::vstack({positions_s, positions_e});
  } else {
    positions = positions_s;
  }

  // cell
  // TODO: now, the box data used by the model file is in unit of Angstrom.
  // We should warn the users about this default
  torch::Tensor cell;
  if (pbc) {
    PLMD::Tensor box = getBox();
    std::vector<float> cell_vector(9);
    std::fill(cell_vector.begin(), cell_vector.end(), 0.0);
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++)
        cell_vector[i * 3 + j] = box[i][j] * to_ang;
    }
    cell = torch::from_blob(
      cell_vector.data(),
      9,
      torch::TensorOptions().dtype(torch::kFloat32)
    );
    cell = cell.to(device).to(torch_float_dtype);
    cell = cell.reshape({3, 3});
  } else {
    cell = torch::zeros({1, 1}, torch::dtype(torch::kFloat32));
    cell = cell.to(device).to(torch_float_dtype);
  }

  // build node attributes
  torch::Tensor node_attrs;
  std::vector<float> node_attrs_vector_s(n_embeddings * n_atoms_padded);
  std::fill(node_attrs_vector_s.begin(), node_attrs_vector_s.end(), 0.0);
  #pragma omp parallel for num_threads(n_threads)
  for (int i = 0; i < n_atoms_a; i++) {
    int index = atom_list_active[i];
    int node_type = system_node_types[getAbsoluteIndex(index).index()];
    node_attrs_vector_s[i * n_embeddings + 0] = node_type;
    if (model_residue_names.size() > 0) {
      int residue_type = system_residue_types[getAbsoluteIndex(index).index()];
      node_attrs_vector_s[i * n_embeddings + 1] = residue_type;
    }
  }
  torch::Tensor node_attrs_s = torch::from_blob(
    node_attrs_vector_s.data(),
    n_embeddings * n_atoms_padded,
    torch::TensorOptions().dtype(torch::kFloat32)
  );
  node_attrs_s = node_attrs_s.to(device).to(torch::kInt64);
  node_attrs_s = node_attrs_s.reshape({n_atoms_padded, n_embeddings});

  if (n_centers > 0) {
    std::vector<float> node_attrs_vector_e(
      n_embeddings * n_atoms_padded_environment
    );
    std::fill(
      node_attrs_vector_e.begin(), node_attrs_vector_e.end(), 0.0
    );
    #pragma omp parallel for num_threads(n_threads)
    for (int i = n_atoms_a; i < n_atoms; i++) {
      int index = atom_list_active[i];
      int node_type = system_node_types[getAbsoluteIndex(index).index()];
      node_attrs_vector_e[(i - n_atoms_a) * n_embeddings + 0] = node_type;
      if (model_residue_names.size() > 0) {
        int residue_type = system_residue_types[getAbsoluteIndex(index).index()];
        node_attrs_vector_e[(i - n_atoms_a) * n_embeddings + 1] = residue_type;
      }
    }
    torch::Tensor node_attrs_e = torch::from_blob(
      node_attrs_vector_e.data(),
      n_embeddings * n_atoms_padded_environment,
      torch::TensorOptions().dtype(torch::kFloat32)
    );
    node_attrs_e = node_attrs_e.to(device).to(torch::kInt64);
    node_attrs_e = node_attrs_e.reshape(
      {n_atoms_padded_environment, n_embeddings}
    );
    node_attrs = torch::vstack({node_attrs_s, node_attrs_e});
  } else {
    node_attrs = node_attrs_s;
  }

  // other things
  // TODO: some of these things are required by MACE. We should disable
  // some of them when not using MACE, maybe by distinguishing the MACE model.
  auto ptr = torch::empty({2}, torch::dtype(torch::kInt64));
  auto weight = torch::empty({1}, torch_float_dtype);
  auto label = torch::ones({1, 1}, torch_float_dtype);
  auto n_system_padded = torch::ones({1, 1}, torch_float_dtype);
  auto n_environment_padded = torch::ones({1, 1}, torch_float_dtype);
  ptr[0] = 0;
  ptr[1] = n_atoms_padded + n_atoms_padded_environment;
  weight[0] = 1.0;
  n_system_padded[0][0] = n_atoms_padded;
  n_environment_padded[0][0] = n_atoms_padded_environment;

  auto system_masks_padded = torch::vstack({
    torch::ones(
      {n_atoms_padded, 1}, torch::dtype(torch::kBool)
    ),
    torch::zeros(
      {n_atoms_padded_environment, 1}, torch::dtype(torch::kBool)
    ),
  });
  auto environment_masks = torch::vstack({
    torch::ones(
      {n_neighbors, 1}, torch::dtype(torch::kBool)
    ),
    torch::zeros(
      {n_atoms_padded_environment - n_neighbors, 1}, torch::dtype(torch::kBool)
    ),
  });
  auto gradient_masks =  torch::vstack({
    torch::ones(
      {n_atoms_a, 1}, torch::dtype(torch::kBool)
    ),
    torch::zeros(
      {n_atoms_padded - n_atoms_a, 1}, torch::dtype(torch::kBool)
    ),
    torch::ones(
      {n_neighbors, 1}, torch::dtype(torch::kBool)
    ),
    torch::zeros(
      {n_atoms_padded_environment - n_neighbors, 1}, torch::dtype(torch::kBool)
    ),
  });

  // load data to device
  // TODO: some of these things are required by MACE. We should disable
  // some of them when not using MACE, maybe by distinguishing the MACE model.
  ptr = ptr.to(device);
  weight = weight.to(device);
  label = label.to(device);
  n_system_padded = n_system_padded.to(device);
  n_environment_padded = n_environment_padded.to(device);
  system_masks_padded = system_masks_padded.to(device);
  environment_masks = environment_masks.to(device);
  gradient_masks = gradient_masks.to(device);

  // require gradients of positions
  positions.requires_grad_(true);

  // NOTE: pack inputs, see: mlcolvar.pairformer.utils.export._dict_to_tensors_pair()

  std::vector<torch::Tensor> input_vector = {
    positions,
    node_attrs,
    cell,
    weight,
    label,
    ptr,
    pair_masks.clone(),
    n_system_padded,
    system_masks_padded,
    n_environment_padded,
    environment_masks,
    centers,
    residue_adjustency,
    system_masks,
  };

  // forward
  std::vector<torch::Tensor> outputs = model->run(input_vector);

  std::vector<PLMD::Vector> derivatives(n_atoms);

  if (!is_committor) {
    for (int i = 0; i < n_cvs; i++) {
      // set CV values
      string name_comp = "node-" + std::to_string(i);
      getPntrToComponent(name_comp)->set(
        outputs[0][0][i].cpu().item<double>()
      );
      // set derivatives
      auto gradients = torch::masked_select(outputs[1][i], gradient_masks);
      gradients = gradients.reshape({n_atoms, 3}).cpu();
      clip_gradients(gradients);
      #pragma omp parallel for num_threads(n_threads)
      for (int j = 0; j < n_atoms; j++) {
        derivatives[j][0] = gradients[j][0].item<double>() * to_ang;
        derivatives[j][1] = gradients[j][1].item<double>() * to_ang;
        derivatives[j][2] = gradients[j][2].item<double>() * to_ang;
      }
      #pragma omp parallel for num_threads(n_threads)
      for (int j = 0; j < n_atoms; j++) {
        int index = atom_list_active[j];
        setAtomsDerivatives(
          getPntrToComponent(name_comp), index, derivatives[j]
        );
      }
    }
  } else if (!k_bias) {
    // set committor values
    string name_comp_z = "node-0";
    getPntrToComponent(name_comp_z)->set(
      outputs[0][0][0].cpu().item<double>()
    );
    string name_comp_q = "node-1";
    getPntrToComponent(name_comp_q)->set(
      outputs[0][0][1].cpu().item<double>()
    );
    // set derivatives of z
    auto gradients = torch::masked_select(outputs[1][0], gradient_masks);
    gradients = gradients.reshape({n_atoms, 3}).cpu();
    clip_gradients(gradients);
    #pragma omp parallel for num_threads(n_threads)
    for (int j = 0; j < n_atoms; j++) {
      derivatives[j][0] = gradients[j][0].item<double>() * to_ang;
      derivatives[j][1] = gradients[j][1].item<double>() * to_ang;
      derivatives[j][2] = gradients[j][2].item<double>() * to_ang;
    }
    #pragma omp parallel for num_threads(n_threads)
    for (int j = 0; j < n_atoms; j++) {
      int index = atom_list_active[j];
      setAtomsDerivatives(
        getPntrToComponent(name_comp_z), index, derivatives[j]
      );
    }
  } else {
    // set committor values
    string name_comp_z = "node-0";
    getPntrToComponent(name_comp_z)->set(
      outputs[0][0][0].cpu().item<double>()
    );
    string name_comp_q = "node-1";
    getPntrToComponent(name_comp_q)->set(
      outputs[0][0][1].cpu().item<double>()
    );
    string name_comp_b = "kbias";
    getPntrToComponent(name_comp_b)->set(
      outputs[2].cpu().item<double>()
    );
    // set derivatives of z
    auto gradients_z = torch::masked_select(outputs[1][0], gradient_masks);
    gradients_z = gradients_z.reshape({n_atoms, 3}).cpu();
    clip_gradients(gradients_z);
    #pragma omp parallel for num_threads(n_threads)
    for (int j = 0; j < n_atoms; j++) {
      derivatives[j][0] = gradients_z[j][0].item<double>() * to_ang;
      derivatives[j][1] = gradients_z[j][1].item<double>() * to_ang;
      derivatives[j][2] = gradients_z[j][2].item<double>() * to_ang;
    }
    #pragma omp parallel for num_threads(n_threads)
    for (int j = 0; j < n_atoms; j++) {
      int index = atom_list_active[j];
      setAtomsDerivatives(
        getPntrToComponent(name_comp_z), index, derivatives[j]
      );
    }
    // set derivatives of bias
    auto gradients_b = torch::masked_select(outputs[3][0], gradient_masks);
    gradients_b = gradients_b.reshape({n_atoms, 3}).cpu();
    clip_gradients(gradients_b);
    #pragma omp parallel for num_threads(n_threads)
    for (int j = 0; j < n_atoms; j++) {
      derivatives[j][0] = gradients_b[j][0].item<double>() * to_ang;
      derivatives[j][1] = gradients_b[j][1].item<double>() * to_ang;
      derivatives[j][2] = gradients_b[j][2].item<double>() * to_ang;
    }
    #pragma omp parallel for num_threads(n_threads)
    for (int j = 0; j < n_atoms; j++) {
      int index = atom_list_active[j];
      setAtomsDerivatives(
        getPntrToComponent(name_comp_b), index, derivatives[j]
      );
    }
  }
}


void PytorchPairFormerExported::check_embedding_is_implemented(std::string name) {
  auto iter = std::find(
    implemented_embeddings.begin(), implemented_embeddings.end(), name
  );
  if (iter == implemented_embeddings.end())
    plumed_merror("Extra embedding '" + name + "' is not implemented!");
}


void PytorchPairFormerExported::find_active_atoms(int n_threads) {
  int n_atoms_a = (int)atom_list_a.size();

  if (atom_list_b.size() > 0) {

    atom_list_active.clear();
    std::vector<int> neighbors(neighbor_list->size());

    #pragma omp parallel for num_threads(n_threads)
    for (size_t i = 0; i < neighbor_list->size(); i++)
      neighbors[i] = neighbor_list->getClosePair(i).second;

    // TODO: make this faster
    std::unordered_set<int> neighbors_set;
    for (int i : neighbors)
      neighbors_set.insert(i);
    neighbors.assign(neighbors_set.begin(), neighbors_set.end());

    // check NL size
    int n_neighbors = (int)neighbors.size();
    if (n_atoms_padded_environment < n_neighbors) {
      log.printf(
        "NL size (%d) is larger than parameter `n_atoms_padded_environment` (%d).\n",
        n_neighbors,
        n_atoms_padded_environment
      );
      if (ignore_excessed_nl_atoms) {
        log.printf(
          "%d neighbor atoms will be ignored. If you see this message too often, re-export your model with a larger `n_atoms_padded_environment` parameter.\n",
          n_neighbors - n_atoms_padded_environment
        );
        n_neighbors = n_atoms_padded_environment;
      } else {
        plumed_merror(
          "Re-export your model with a larger `n_atoms_padded_environment` parameter!"
        );
      }
    }

    // NOTE: the system atoms (atom_list_a) should always appear at the head of
    // this list. Do NOT change the order!
    for (int i = 0; i < n_atoms_a; i++)
      atom_list_active.push_back(i);
    // NOTE: the neighbors should be appended to the tail of this list.
    // besides, the first `n_atoms_a` elements of `x_local` belong to system
    // atoms, thus we plus an offset of `n_atoms_a` to neighbor indices.
    for (int i = 0; i < n_neighbors; i++)
      atom_list_active.push_back(neighbors[i] + n_atoms_a);
  } else if (atom_list_active.size() == 0) {
    atom_list_active.clear();

    for (int i = 0; i < n_atoms_a; i++)
      atom_list_active.push_back(i);
  }
}


bool PytorchPairFormerExported::groups_have_intersection(void) {
  std::vector<AtomNumber> intersections;
  std::vector<AtomNumber> atom_list_a_copy(atom_list_a);
  std::vector<AtomNumber> atom_list_b_copy(atom_list_b);

  std::sort(atom_list_a_copy.begin(), atom_list_a_copy.end());
  std::sort(atom_list_b_copy.begin(), atom_list_b_copy.end());

  std::set_intersection(
    atom_list_a_copy.begin(),
    atom_list_a_copy.end(),
    atom_list_b_copy.begin(),
    atom_list_b_copy.end(),
    back_inserter(intersections)
  );

  return intersections.size() > 0;
}


bool PytorchPairFormerExported::group_c_is_in_group_a(void) {
  std::vector<AtomNumber> atom_list_a_copy(atom_list_a);
  std::vector<std::vector<AtomNumber>> atom_lists_c_copy(atom_lists_c);
  for (auto list: atom_lists_c_copy) {
    for (auto atom_elt: list)
      if (
        std::find(atom_list_a_copy.begin(), atom_list_a_copy.end(), atom_elt)
        == atom_list_a_copy.end()
      )
        return false;
  }
  return true;
}


std::string PytorchPairFormerExported::trim(const std::string& str)
{
    size_t first = str.find_first_not_of(' ');
    if (std::string::npos == first)
        return str;
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}


void PytorchPairFormerExported::clip_gradients(torch::Tensor &gradients)
{
  torch::Tensor mask = torch::abs(gradients) > gradient_threshold;
  mask = mask.to(torch_float_dtype);
  if (mask.sum().item<double>() > 0) {
    log.printf(
      (
        "Maximum gradient (%.6e) is larger than parameter `gradient_threshold` (%.6e) and thus is clipped to the later value. "
        "If you see this message too often, use a larger `gradient_threshold` parameter.\n\n"
      ),
      torch::max(torch::abs(gradients)).item<double>(),
      gradient_threshold
    );
    gradients = (
      gradients * (1.0 - mask)
      + mask * gradient_threshold * torch::sign(gradients)
    );
  }
}


} // pytorch_pairformer_exported

} // colvar

} // PLMD

#endif // PLUMED_HAS_LIBTORCH
