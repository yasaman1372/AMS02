#ifndef PhotonTree_hh
#define PhotonTree_hh

#include "TreeInterface.hh"
#include "Utilities.hh"
#include "BinningDefinition.hh"
#include "BinningTools.hh"

namespace Analysis {
  class Track;
  class TrackFactory;
  class TrdHitFactory;
};

class TH1;

class AMSEventR;
class TrTrackR;

class PhotonTree : public IO::TreeInterface {
public:
  PhotonTree();

  IO::TreeBranch<unsigned int> Time { "Time", 0, {Binning::Equidistant(10000, 1.307e9, 1.62e9)}};
  IO::TreeBranch<double> UTCTime { "UTCTime", 0, {Binning::Equidistant(10000, 1.307e9, 1.62e9)}};
  IO::TreeBranch<unsigned int> RunNumber { "RunNumber", 0 };
  IO::TreeBranch<unsigned int> EventNumber { "EventNumber", 0 };
  IO::TreeBranch<float> McWeight { "McWeight", 1.0f };
  IO::TreeBranch<float> PrescalingWeight { "PrescalingWeight", 1.0f };
  IO::TreeBranch<uint8_t> PhotonTreeVersion { "PhotonTreeVersion", 0 };

  IO::TreeBranch<float> TriggerRateFT { "TriggerRateFT", -100., {Binning::Equidistant(200, 0, 5000.)}};
  IO::TreeBranch<float> TriggerRateL1 { "TriggerRateL1", -100., {Binning::Equidistant(200, 0, 5000.)}};
  IO::TreeBranch<unsigned char> TriggerFlags { "TriggerFlags", 0, {Binning::Equidistant(100, -0.5, 99.5), "flag", "Trigger flag"}};

  // Coordinate System Transformation
  IO::TreeBranch<std::vector<float>> ISSParameters { "ISSParameters", IO::TreeVectorSize(9) };

  // MC
  IO::TreeBranch<short> McParticleID { "McParticleID", 0 };
  IO::TreeBranch<float> McMomentum { "McMomentum", 0.0f };
  IO::TreeBranch<float> McPrimaryX { "McPrimaryX", -1000.0 };
  IO::TreeBranch<float> McPrimaryY { "McPrimaryY", -1000.0 };
  IO::TreeBranch<float> McPrimaryZ { "McPrimaryZ", -1000.0 };
  IO::TreeBranch<std::vector<float>> McPrimaryDirXYZ { "McPrimaryDirXYZ", IO::TreeVectorSize(3) };
  IO::TreeBranch<float> McPrimaryFinalMomentum { "McPrimaryFinalMomentum", 0.0f };
  IO::TreeBranch<float> McPrimaryFinalX { "McPrimaryFinalX", 0.0f };
  IO::TreeBranch<float> McPrimaryFinalY { "McPrimaryFinalY", 0.0f };
  IO::TreeBranch<float> McPrimaryFinalZ { "McPrimaryFinalZ", 0.0f };
  IO::TreeBranch<std::vector<float>> McPrimaryFinalDirXYZ { "McPrimaryFinalDirXYZ", IO::TreeVectorSize(3) };
  IO::TreeBranch<std::vector<short>> McSecondaryCreatingProcesses { "McSecondaryCreatingProcesses", IO::TreeVectorSize(2) };
  IO::TreeBranch<std::vector<short>> McSecondaryParticleIDs { "McSecondaryParticleIDs", IO::TreeVectorSize(2) };
  IO::TreeBranch<std::vector<float>> McSecondaryMomenta { "McSecondaryMomenta", IO::TreeVectorSize(2) };
  IO::TreeBranch<std::vector<float>> McSecondaryPositions { "McSecondaryPositions", IO::TreeVectorSize(6) };
  IO::TreeBranch<std::vector<float>> McSecondaryDirectionsXYZ { "McSecondaryDirectionsXYZ", IO::TreeVectorSize(6) };
  IO::TreeBranch<std::vector<float>> McSecondaryBremsstrahlungEnergyLossTotal { "McSecondaryBremsstrahlungEnergyLossTotal", IO::TreeVectorSize(2) };
  IO::TreeBranch<std::vector<float>> McSecondaryBremsstrahlungEnergyLossBeforeL3 { "McSecondaryBremsstrahlungEnergyLossBeforeL3", IO::TreeVectorSize(2) };
  IO::TreeBranch<std::vector<float>> McSecondaryBremsstrahlungEnergyLossBeforeL56 { "McSecondaryBremsstrahlungEnergyLossBeforeL56", IO::TreeVectorSize(2) };
  IO::TreeBranch<std::vector<float>> McSecondaryBremsstrahlungEnergyLossBeforeL8 { "McSecondaryBremsstrahlungEnergyLossBeforeL8", IO::TreeVectorSize(2) };

  IO::TreeBranch<float> McTrackZMin { "McTrackZMin", 0.0f };
  IO::TreeBranch<float> McTrackZMax { "McTrackZMax", 0.0f };
  IO::TreeBranch<std::vector<float>> McTrackCoordX { "McTrackCoordX", IO::TreeVectorSize(9) };
  IO::TreeBranch<std::vector<float>> McTrackCoordY { "McTrackCoordY", IO::TreeVectorSize(9) };
  IO::TreeBranch<std::vector<float>> McTrackCoordZ { "McTrackCoordZ", IO::TreeVectorSize(9) };
  IO::TreeBranch<std::vector<float>> McTrackDirXYZ { "McTrackDirXYZ", IO::TreeVectorSize(9) };

  // TRD
  IO::TreeBranch<short> TrdNActiveLayers { "TrdNActiveLayers", -1 };
  IO::TreeBranch<short> NTrdHits { "NTrdHits", 0 };
  IO::TreeBranch<std::vector<short>> TrdHitsInLayer { "TrdHitsInLayer", IO::TreeVectorSize(20) };
  IO::TreeBranch<std::vector<float>> TrdHitCoordinates { "TrdHitCoordinates", IO::TreeVectorSize(60) };
  IO::TreeBranch<std::vector<float>> TrdHitEnergies { "TrdHitEnergies", IO::TreeVectorSize(20) };
  IO::TreeBranch<std::vector<short>> TrdHitLayers { "TrdHitLayers", IO::TreeVectorSize(20) };

  IO::TreeBranch<std::vector<short>> TrdSegmentsXZNHits { "TrdSegmentsXZNHits", IO::TreeVectorSize(2) };
  IO::TreeBranch<std::vector<uint32_t>> TrdSegmentsXZSublayerPattern { "TrdSegmentsXZSublayerPattern", IO::TreeVectorSize(2) };
  IO::TreeBranch<std::vector<float>> TrdSegmentsXZFitParameters { "TrdSegmentsXZFitParameters", IO::TreeVectorSize(14) };
  IO::TreeBranch<std::vector<short>> TrdSegmentsYZNHits { "TrdSegmentsYZNHits", IO::TreeVectorSize(2) };
  IO::TreeBranch<std::vector<uint32_t>> TrdSegmentsYZSublayerPattern { "TrdSegmentsYZSublayerPattern", IO::TreeVectorSize(2) };
  IO::TreeBranch<std::vector<float>> TrdSegmentsYZFitParameters { "TrdSegmentsYZFitParameters", IO::TreeVectorSize(14) };

  // ACC
  IO::TreeBranch<short> AccNClusters { "AccNClusters", -1, {Binning::Equidistant(10, -0.5, 9.5)}};
  IO::TreeBranch<short> AccNClustersTrigger { "AccNClustersTrigger", -1, {Binning::Equidistant(10, -0.5, 9.5)}};
  IO::TreeBranch<std::vector<float>> AccClustersZ { "AccClustersZ", IO::TreeVectorSize(2) };
  IO::TreeBranch<std::vector<float>> AccClustersPhi { "AccClustersPhi", IO::TreeVectorSize(2) };
  IO::TreeBranch<std::vector<float>> AccClustersEnergy { "AccClustersEnergy", IO::TreeVectorSize(2) };
  IO::TreeBranch<std::vector<int>> AccClustersNumberOfPairs { "AccClustersNumberOfPairs", IO::TreeVectorSize(2) };

  // TOF
  IO::TreeBranch<std::vector<short>> TofClustersInLayer { "TofClustersInLayer", IO::TreeVectorSize(4) };
  IO::TreeBranch<std::vector<float>> TofClusterCoordinates { "TofClusterCoordinates", IO::TreeVectorSize(18) };
  IO::TreeBranch<std::vector<float>> TofClusterEnergies { "TofClusterEnergies", IO::TreeVectorSize(6) };
  IO::TreeBranch<std::vector<float>> TofClusterCharges { "TofClusterCharges", IO::TreeVectorSize(6) };
  IO::TreeBranch<std::vector<short>> TofClusterLayers { "TofClusterLayers", IO::TreeVectorSize(6) };

  // Tracker
  IO::TreeBranch<short> TrkNTracks { "TrkNTracks", 0, {Binning::UpToNBinning(10)}};
  IO::TreeBranch<std::vector<float>> TrkTrackCharges { "TrkTrackCharges", IO::TreeVectorSize(4) };
  IO::TreeBranch<std::vector<float>> TrkTrackLayerCharges { "TrkTrackLayerCharges", IO::TreeVectorSize(36) };
  IO::TreeBranch<std::vector<float>> TrkTrackChargeErrors { "TrkTrackChargeErrors", IO::TreeVectorSize(4) };
  IO::TreeBranch<std::vector<short>> TrkTrackHits { "TrkTrackHits", IO::TreeVectorSize(4) };
  IO::TreeBranch<std::vector<short>> TrkTrackLayerPattern { "TrkTrackLayerPattern", IO::TreeVectorSize(4) };
  IO::TreeBranch<std::vector<float>> TrkTrackRigidities { "TrkTrackRigidities", IO::TreeVectorSize(4) };
  IO::TreeBranch<std::vector<float>> TrkTrackRigidityChiSquaresY { "TrkTrackRigidityChiSquaresY", IO::TreeVectorSize(4) };
  IO::TreeBranch<std::vector<float>> TrkTrackRigidityChiSquaresX { "TrkTrackRigidityChiSquaresX", IO::TreeVectorSize(4) };
  IO::TreeBranch<std::vector<float>> TrkTrackInverseRigidityErrors { "TrkTrackInverseRigidityErrors", IO::TreeVectorSize(4) };
  IO::TreeBranch<std::vector<float>> TrkTrackFitResidualsL1Y { "TrkTrackFitResidualsL1Y", IO::TreeVectorSize(4) };
  IO::TreeBranch<std::vector<float>> TrkTrackElectronRigidities { "TrkTrackElectronRigidities", IO::TreeVectorSize(4) };
  IO::TreeBranch<std::vector<float>> TrkTrackElectronRigidityChiSquaresY { "TrkTrackElectronRigidityChiSquaresY", IO::TreeVectorSize(4) };
  IO::TreeBranch<std::vector<float>> TrkTrackElectronRigidityChiSquaresX { "TrkTrackElectronRigidityChiSquaresX", IO::TreeVectorSize(4) };
  IO::TreeBranch<std::vector<float>> TrkTrackInverseElectronRigidityErrors { "TrkTrackInverseElectronRigidityErrors", IO::TreeVectorSize(4) };
  IO::TreeBranch<std::vector<float>> TrkTrackElectronFitResidualsL1Y { "TrkTrackElectronFitResidualsL1Y", IO::TreeVectorSize(4) };
  IO::TreeBranch<std::vector<float>> TrkTrackHitCoordinates { "TrkTrackHitCoordinates", IO::TreeVectorSize(27) };
  IO::TreeBranch<std::vector<float>> TrkTrackFitCoordinates { "TrkTrackFitCoordinates", IO::TreeVectorSize(27) };
  IO::TreeBranch<std::vector<float>> TrkTrackFitDirectionsXYZ { "TrkTrackFitDirectionsXYZ", IO::TreeVectorSize(27) };
  IO::TreeBranch<std::vector<float>> TrkTrackCoordinatesAtUpperTof { "TrkTrackCoordinatesAtUpperTof", IO::TreeVectorSize(8) };
  IO::TreeBranch<std::vector<float>> TrkTrackDirectionsAtUpperTofXYZ { "TrkTrackDirectionsAtUpperTofXYZ", IO::TreeVectorSize(8) };
  IO::TreeBranch<std::vector<float>> TrkTrackCoordinatesAtLowerTof { "TrkTrackCoordinatesAtLowerTof", IO::TreeVectorSize(8) };
  IO::TreeBranch<std::vector<float>> TrkTrackDirectionsAtLowerTofXYZ { "TrkTrackDirectionsAtLowerTofXYZ", IO::TreeVectorSize(8) };
  IO::TreeBranch<std::vector<float>> TrkTrackCoordinatesAtLowerTrd { "TrkTrackCoordinatesAtLowerTrd", IO::TreeVectorSize(8) };
  IO::TreeBranch<std::vector<float>> TrkTrackDirectionsAtLowerTrdXYZ { "TrkTrackDirectionsAtLowerTrdXYZ", IO::TreeVectorSize(8) };
  IO::TreeBranch<std::vector<float>> TrkTrackCoordinatesAtECAL { "TrkTrackCoordinatesAtECAL", IO::TreeVectorSize(8) };
  IO::TreeBranch<std::vector<float>> TrkTrackDirectionsAtECALXYZ { "TrkTrackDirectionsAtECALXYZ", IO::TreeVectorSize(8) };
  IO::TreeBranch<std::vector<float>> TrkTrackAssociatedTofBetas { "TrkTrackAssociatedTofBetas", IO::TreeVectorSize(4) };
  IO::TreeBranch<std::vector<short>> TrkTrackAssociatedTofClusterIndices { "TrkTrackAssociatedTofClusterIndices", IO::TreeVectorSize(16) };

  IO::TreeBranch<std::vector<short>> TrkTrackPairTrackIndices { "TrkTrackPairTrackIndices", IO::TreeVectorSize(16) };
  IO::TreeBranch<std::vector<float>> TrkTrackPairMinDistances { "TrkTrackPairMinDistances", IO::TreeVectorSize(8) };
  IO::TreeBranch<std::vector<float>> TrkTrackPairMinDistanceAngles { "TrkTrackPairMinDistanceAngles", IO::TreeVectorSize(8) };
  IO::TreeBranch<std::vector<float>> TrkTrackPairMinDistanceCoordinates { "TrkTrackPairMinDistanceCoordinates", IO::TreeVectorSize(24) };
  IO::TreeBranch<std::vector<float>> TrkTrackPairMinDistanceDirectionsXYZ { "TrkTrackPairMinDistanceDirectionsXYZ", IO::TreeVectorSize(24) };

  IO::TreeBranch<std::vector<float>> TrkVertexCoordinates { "TrkVertexCoordinates", IO::TreeVectorSize(5) };
  IO::TreeBranch<std::vector<float>> TrkVertexMomentum { "TrkVertexMomentum", IO::TreeVectorSize(1) };
  IO::TreeBranch<std::vector<short>> TrkVertexTrackIndices { "TrkVertexTrackIndices", IO::TreeVectorSize(2) };

  IO::TreeBranch<std::vector<float>> TrkMaxLayerCharges { "TrkMaxLayerCharges", IO::TreeVectorSize(9) };

  IO::TreeBranch<std::vector<short>> TrkHitLayers { "TrkHitLayers", IO::TreeVectorSize(9) };
  IO::TreeBranch<std::vector<float>> TrkHitCoords { "TrkHitCoords", IO::TreeVectorSize(27) };
  IO::TreeBranch<std::vector<float>> TrkHitCharges { "TrkHitCharges", IO::TreeVectorSize(18) };
  IO::TreeBranch<std::vector<short>> TrkHitTrackIndices { "TrkHitTrackIndices", IO::TreeVectorSize(9) };
  IO::TreeBranch<std::vector<short>> TrkHitClusterIndicesX { "TrkHitClusterIndicesX", IO::TreeVectorSize(9) };
  IO::TreeBranch<std::vector<short>> TrkHitClusterIndicesY { "TrkHitClusterIndicesY", IO::TreeVectorSize(9) };

  IO::TreeBranch<std::vector<uint8_t>> TrkClusterLayersX { "TrkClusterLayersX", IO::TreeVectorSize(9) };
  IO::TreeBranch<std::vector<short>> TrkClusterLengthsX { "TrkClusterLengthsX", IO::TreeVectorSize(9) };
  IO::TreeBranch<std::vector<float>> TrkClusterAmplitudesX { "TrkClusterAmplitudesX", IO::TreeVectorSize(27) };
  IO::TreeBranch<std::vector<short>> TrkClusterOffsetsX { "TrkClusterOffsetsX", IO::TreeVectorSize(9) };
  IO::TreeBranch<std::vector<uint8_t>> TrkClusterLayersY { "TrkClusterLayersY", IO::TreeVectorSize(9) };
  IO::TreeBranch<std::vector<short>> TrkClusterLengthsY { "TrkClusterLengthsY", IO::TreeVectorSize(9) };
  IO::TreeBranch<std::vector<float>> TrkClusterAmplitudesY { "TrkClusterAmplitudesY", IO::TreeVectorSize(27) };
  IO::TreeBranch<std::vector<short>> TrkClusterOffsetsY { "TrkClusterOffsetsY", IO::TreeVectorSize(9) };
  IO::TreeBranch<std::vector<float>> TrkClusterCoordinatesY { "TrkClusterCoordinatesY", IO::TreeVectorSize(9) };

  // Geomagnetic Cutoff
  IO::TreeBranch<float> Longitude { "Longitude", -1000.0, {Binning::Equidistant(200, -200.0f, 200.0f)}};
  IO::TreeBranch<float> Latitude { "Latitude", -1000.0, {Binning::Equidistant(200, -100.0f, 100.0f)}};

  // Ecal
  IO::TreeBranch<short> NEcalShowers { "NEcalShowers", 0 };
  IO::TreeBranch<float> EcalEnergy { "EcalEnergy", 0.0f };
  IO::TreeBranch<float> EcalBdt { "EcalBdt", 0.0f };
  IO::TreeBranch<float> EcalIntegralLikelihood { "EcalIntegralLikelihood", 0.0f };
  IO::TreeBranch<float> EcalReweightedLikelihood { "EcalReweightedLikelihood", 0.0f };
  IO::TreeBranch<std::vector<float>> EcalShowerEnergies { "EcalShowerEnergies", IO::TreeVectorSize(3) };
  IO::TreeBranch<std::vector<float>> EcalShowerPositions { "EcalShowerPositions", IO::TreeVectorSize(9) };
  IO::TreeBranch<std::vector<float>> EcalShowerDirectionsXYZ { "EcalShowerDirectionsXYZ", IO::TreeVectorSize(9) };

  IO::TreeBranch<short> NEcal2DShowers { "NEcal2DShowers", 0 };
  IO::TreeBranch<std::vector<float>> Ecal2DShowerEnergies { "Ecal2DShowerEnergies", IO::TreeVectorSize(3) };
  IO::TreeBranch<std::vector<float>> Ecal2DShowerDepositedEnergies { "Ecal2DShowerDepositedEnergies", IO::TreeVectorSize(3) };
  IO::TreeBranch<std::vector<float>> Ecal2DShowerPositions { "Ecal2DShowerPositions", IO::TreeVectorSize(9) };
  IO::TreeBranch<std::vector<float>> Ecal2DShowerEntryPositions { "Ecal2DShowerEntryPositions", IO::TreeVectorSize(9) };
  IO::TreeBranch<std::vector<float>> Ecal2DShowerDirectionsCoGXYZ { "Ecal2DShowerDirectionsCoGXYZ", IO::TreeVectorSize(6) };
  IO::TreeBranch<std::vector<float>> Ecal2DShowerDirectionsCRXYZ { "Ecal2DShowerDirectionsCRXYZ", IO::TreeVectorSize(6) };
  IO::TreeBranch<std::vector<float>> Ecal2DShowerDirectionsEMXYZ { "Ecal2DShowerDirectionsEMXYZ", IO::TreeVectorSize(6) };
  IO::TreeBranch<std::vector<float>> Ecal2DShowerBdts { "Ecal2DShowerBdts", IO::TreeVectorSize(3) };
  IO::TreeBranch<std::vector<float>> Ecal2DShowerLongitudinalChiSquares { "Ecal2DShowerLongitudinalChiSquares", IO::TreeVectorSize(3) };
  IO::TreeBranch<std::vector<float>> Ecal2DShowerLateralChiSquares { "Ecal2DShowerLateralChiSquares", IO::TreeVectorSize(3) };
  IO::TreeBranch<std::vector<float>> Ecal2DShowerRatio1cm { "Ecal2DShowerRatio1cm", IO::TreeVectorSize(3) };
  IO::TreeBranch<std::vector<float>> Ecal2DShowerRatio3cm { "Ecal2DShowerRatio3cm", IO::TreeVectorSize(3) };
  IO::TreeBranch<std::vector<uint32_t>> Ecal2DShowerStatus { "Ecal2DShowerStatus", IO::TreeVectorSize(3) };

  float CalculatePrescalingWeight(const Analysis::Event&);

  bool fWithAmsVariables;
  bool fWithTrackerHits;
  bool fPrescaleTrdHits;
  bool fPrescaleSameSign;
  bool fPrescaleSingleTrack;
  bool fPrescaleHadronicShower; 

private:
  virtual void Fill(const Analysis::Event&);
  virtual IO::TreeInterface* Create() const { return new PhotonTree; }
  //virtual const IO::TreeBranchBase<UInt_t>* CurrentEventTime() const { return &Time; }
  //virtual const IO::TreeBranchBase<UChar_t>* CurrentTriggerFlags() const { return &TriggerFlags; }
  //virtual const IO::TreeBranchBase<Double_t>* CurrentWeight() const { return &TotalWeight(); }
  //virtual const IO::TreeBranchBase<UInt_t>* CurrentRunNumber() const { return &RunNumber; }
  //virtual const IO::TreeBranchBase<UInt_t>* CurrentEventNumber() const { return &EventNumber; }

  Analysis::TrackFactory* fTrackFactory;
  Analysis::TrdHitFactory* fTrdHitFactory;

};

#endif
