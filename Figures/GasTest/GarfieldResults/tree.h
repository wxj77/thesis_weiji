//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Sun Jul 22 23:41:28 2018 by ROOT version 5.34/30
// from TTree tree/gardield sim tree
// found on file: result_1.0_2000_xxx.root
//////////////////////////////////////////////////////////

#ifndef tree_h
#define tree_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.

// Fixed size dimensions of array or collections stored in the TTree if any.

class tree {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leaf types
   Int_t           ne;
   Int_t           ni;
   Int_t           nexc;
   Int_t           NE;
   Int_t           NI;
   Int_t           NEXC;
   Float_t         DURATION;
   Float_t         EFRONT;
   Float_t         EBACK;
   Int_t           TY[446];   //[NEXC]
   Float_t         XX[446];   //[NEXC]
   Float_t         YY[446];   //[NEXC]
   Float_t         ZZ[446];   //[NEXC]
   Float_t         TT[446];   //[NEXC]
   Float_t         EF[446];   //[NEXC]
   Float_t         EX[446];   //[NEXC]
   Float_t         EY[446];   //[NEXC]
   Float_t         EZ[446];   //[NEXC]
   Int_t           LE[446];   //[NEXC]
   Int_t           TYi[4];   //[NI]
   Float_t         XXi[4];   //[NI]
   Float_t         YYi[4];   //[NI]
   Float_t         ZZi[4];   //[NI]
   Float_t         TTi[4];   //[NI]
   Float_t         EFi[4];   //[NI]
   Float_t         EXi[4];   //[NI]
   Float_t         EYi[4];   //[NI]
   Float_t         EZi[4];   //[NI]
   Int_t           LEi[4];   //[NI]
   Float_t         XXXf[5];   //[NE]
   Float_t         YYYf[5];   //[NE]
   Float_t         ZZZf[5];   //[NE]
   Float_t         TTTf[5];   //[NE]
   Float_t         EEEf[5];   //[NE]
   Float_t         XXXb[5];   //[NE]
   Float_t         YYYb[5];   //[NE]
   Float_t         ZZZb[5];   //[NE]
   Float_t         TTTb[5];   //[NE]
   Float_t         EEEb[5];   //[NE]

   // List of branches
   TBranch        *b_ne;   //!
   TBranch        *b_ni;   //!
   TBranch        *b_nexc;   //!
   TBranch        *b_NE;   //!
   TBranch        *b_NI;   //!
   TBranch        *b_NEXC;   //!
   TBranch        *b_DURATION;   //!
   TBranch        *b_EFRONT;   //!
   TBranch        *b_EBACK;   //!
   TBranch        *b_TY;   //!
   TBranch        *b_XX;   //!
   TBranch        *b_YY;   //!
   TBranch        *b_ZZ;   //!
   TBranch        *b_TT;   //!
   TBranch        *b_EF;   //!
   TBranch        *b_EX;   //!
   TBranch        *b_EY;   //!
   TBranch        *b_EZ;   //!
   TBranch        *b_LE;   //!
   TBranch        *b_TYi;   //!
   TBranch        *b_XXi;   //!
   TBranch        *b_YYi;   //!
   TBranch        *b_ZZi;   //!
   TBranch        *b_TTi;   //!
   TBranch        *b_EFi;   //!
   TBranch        *b_EXi;   //!
   TBranch        *b_EYi;   //!
   TBranch        *b_EZi;   //!
   TBranch        *b_LEi;   //!
   TBranch        *b_XXXf;   //!
   TBranch        *b_YYYf;   //!
   TBranch        *b_ZZZf;   //!
   TBranch        *b_TTTf;   //!
   TBranch        *b_EEEf;   //!
   TBranch        *b_XXXb;   //!
   TBranch        *b_YYYb;   //!
   TBranch        *b_ZZZb;   //!
   TBranch        *b_TTTb;   //!
   TBranch        *b_EEEb;   //!

   tree(TTree *tree=0);
   virtual ~tree();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef tree_cxx
tree::tree(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("result_1.0_2000_xxx.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("result_1.0_2000_xxx.root");
      }
      f->GetObject("tree",tree);

   }
   Init(tree);
}

tree::~tree()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t tree::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t tree::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void tree::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("ne", &ne, &b_ne);
   fChain->SetBranchAddress("ni", &ni, &b_ni);
   fChain->SetBranchAddress("nexc", &nexc, &b_nexc);
   fChain->SetBranchAddress("NE", &NE, &b_NE);
   fChain->SetBranchAddress("NI", &NI, &b_NI);
   fChain->SetBranchAddress("NEXC", &NEXC, &b_NEXC);
   fChain->SetBranchAddress("DURATION", &DURATION, &b_DURATION);
   fChain->SetBranchAddress("EFRONT", &EFRONT, &b_EFRONT);
   fChain->SetBranchAddress("EBACK", &EBACK, &b_EBACK);
   fChain->SetBranchAddress("TY", TY, &b_TY);
   fChain->SetBranchAddress("XX", XX, &b_XX);
   fChain->SetBranchAddress("YY", YY, &b_YY);
   fChain->SetBranchAddress("ZZ", ZZ, &b_ZZ);
   fChain->SetBranchAddress("TT", TT, &b_TT);
   fChain->SetBranchAddress("EF", EF, &b_EF);
   fChain->SetBranchAddress("EX", EX, &b_EX);
   fChain->SetBranchAddress("EY", EY, &b_EY);
   fChain->SetBranchAddress("EZ", EZ, &b_EZ);
   fChain->SetBranchAddress("LE", LE, &b_LE);
   fChain->SetBranchAddress("TYi", TYi, &b_TYi);
   fChain->SetBranchAddress("XXi", XXi, &b_XXi);
   fChain->SetBranchAddress("YYi", YYi, &b_YYi);
   fChain->SetBranchAddress("ZZi", ZZi, &b_ZZi);
   fChain->SetBranchAddress("TTi", TTi, &b_TTi);
   fChain->SetBranchAddress("EFi", EFi, &b_EFi);
   fChain->SetBranchAddress("EXi", EXi, &b_EXi);
   fChain->SetBranchAddress("EYi", EYi, &b_EYi);
   fChain->SetBranchAddress("EZi", EZi, &b_EZi);
   fChain->SetBranchAddress("LEi", LEi, &b_LEi);
   fChain->SetBranchAddress("XXXf", XXXf, &b_XXXf);
   fChain->SetBranchAddress("YYYf", YYYf, &b_YYYf);
   fChain->SetBranchAddress("ZZZf", ZZZf, &b_ZZZf);
   fChain->SetBranchAddress("TTTf", TTTf, &b_TTTf);
   fChain->SetBranchAddress("EEEf", EEEf, &b_EEEf);
   fChain->SetBranchAddress("XXXb", XXXb, &b_XXXb);
   fChain->SetBranchAddress("YYYb", YYYb, &b_YYYb);
   fChain->SetBranchAddress("ZZZb", ZZZb, &b_ZZZb);
   fChain->SetBranchAddress("TTTb", TTTb, &b_TTTb);
   fChain->SetBranchAddress("EEEb", EEEb, &b_EEEb);
   Notify();
}

Bool_t tree::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void tree::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t tree::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef tree_cxx
