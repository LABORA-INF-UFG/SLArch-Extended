// ns-3.42 (ns-3-dev) with 5g-lena v3.3.y

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/buildings-module.h"
#include "ns3/config-store-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"
#include <fstream>
#include <string>
#include <map>
#include <cmath>
#include <deque>
#include <vector>
#include <algorithm>
#include <filesystem>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("CttcNrDemo");

uint32_t totalUes = 150;
uint32_t batchSize = 10;
uint32_t currentBatch = 0;
uint32_t currentIteration = 0;
uint32_t totalIterations = 10;
std::vector<uint32_t> seeds;
std::vector<std::ofstream> outputFiles;
std::string resultsDir = "nr_slicing_ul_res/";

struct GlobalUeInfo {
    uint32_t globalUeId;           
    Vector position;               
    uint32_t sliceType;            
    uint32_t assignedNodeId;       
    Ipv4Address ipAddress;         
    uint32_t servingGnbId;         
    double distanceToGnb;         
    
    GlobalUeInfo() : globalUeId(0), sliceType(0), assignedNodeId(0), 
                     servingGnbId(0), distanceToGnb(0.0) {}
};

std::vector<GlobalUeInfo> allUeInfos;
std::map<uint32_t, GlobalUeInfo*> globalUeIdToInfoMap; 
std::map<uint32_t, GlobalUeInfo*> nodeIdToUeInfoMap;   
std::map<Ipv4Address, GlobalUeInfo*> ipToUeInfoMap;    

std::vector<Vector> globalGnbPositions;

struct UeSinrData {
    std::deque<double> samples;
    double sum = 0.0;
    int count = 0;
    double latest = -999.0;
    
    void AddSample(double sinrDb) {        
        if (samples.size() >= 50) {
            sum -= samples.front();
            samples.pop_front();
            count--;
        }
        samples.push_back(sinrDb);
        sum += sinrDb;
        count++;
        latest = sinrDb;
    }
    
    double GetAverage() const {
        if (count > 0) {
            return sum / count;
        }
        return -999.0;
    }
};

struct UePowerData {
    double txPowerDbm;           
    double rxPowerDbm;           
    double totalEnergyJoules;    
    Time lastUpdateTime;         
    std::deque<double> txPowerSamples;  
    std::deque<double> rxPowerSamples;  
    UePowerData() : txPowerDbm(0.0), rxPowerDbm(0.0), totalEnergyJoules(0.0), 
                    lastUpdateTime(Seconds(0)) {}    
    void AddPowerSample(double txPower, double rxPower) {        
        txPowerSamples.push_back(txPower);
        rxPowerSamples.push_back(rxPower);
        if (txPowerSamples.size() > 100) {
            txPowerSamples.pop_front();
        }
        if (rxPowerSamples.size() > 100) {
            rxPowerSamples.pop_front();
        }        
        txPowerDbm = txPower;
        rxPowerDbm = rxPower;
    }
    
    double GetAverageTxPower() const {
        if (txPowerSamples.empty()) return 0.0;
        double sum = 0.0;
        for (double p : txPowerSamples) sum += p;
        return sum / txPowerSamples.size();
    }
    
    double GetAverageRxPower() const {
        if (rxPowerSamples.empty()) return 0.0;
        double sum = 0.0;
        for (double p : rxPowerSamples) sum += p;
        return sum / rxPowerSamples.size();
    }
};

std::map<uint32_t, UeSinrData> ueSinrData;
std::map<uint32_t, UePowerData> uePowerData;  
std::map<uint64_t, uint32_t> imsiToNodeIdMap;

bool enableSinrTracingConfig = true;
double sinrSamplingRate = 0.1;
Ptr<UniformRandomVariable> sinrSamplingRandom;

void OptimizedSinrCallback(uint64_t imsi, uint16_t cellId, uint16_t rnti, double sinrLinear, uint8_t componentCarrierId) {
    if (!enableSinrTracingConfig) return;
    
    if (sinrSamplingRandom->GetValue() > sinrSamplingRate) {
        return;
    }
    
    double sinrDb = 10 * log10(sinrLinear);
    
    auto it = imsiToNodeIdMap.find(imsi);
    if (it != imsiToNodeIdMap.end()) {
        uint32_t nodeId = it->second;
        ueSinrData[nodeId].AddSample(sinrDb);
    }
}

double GetAverageSinr(uint32_t nodeId) {
    auto it = ueSinrData.find(nodeId);
    if (it != ueSinrData.end()) {
        return it->second.GetAverage();
    }
    return -999.0;
}

double GetLatestSinr(uint32_t nodeId) {
    auto it = ueSinrData.find(nodeId);
    if (it != ueSinrData.end()) {
        return it->second.latest;
    }
    return -999.0;
}

double GetTotalEnergyConsumption(uint32_t nodeId) {
    auto it = uePowerData.find(nodeId);
    if (it != uePowerData.end()) {
        return it->second.totalEnergyJoules;
    }
    return 0.0;
}

double GetAverageTxPower(uint32_t nodeId) {
    auto it = uePowerData.find(nodeId);
    if (it != uePowerData.end()) {
        return it->second.GetAverageTxPower();
    }
    return 0.0;
}

double GetAverageRxPower(uint32_t nodeId) {
    auto it = uePowerData.find(nodeId);
    if (it != uePowerData.end()) {
        return it->second.GetAverageRxPower();
    }
    return 0.0;
}

void InitializeGlobalUeInfo() {
    NS_LOG_INFO("Initializing global UE information for " << totalUes << " UEs");
    
    allUeInfos.clear();
    globalUeIdToInfoMap.clear();
    nodeIdToUeInfoMap.clear();
    ipToUeInfoMap.clear();
    
    uint16_t gNbNum = 2;
    globalGnbPositions.clear();
    for (uint32_t i = 0; i < gNbNum; ++i) {
        double xPos = i * 100.0;
        globalGnbPositions.push_back(Vector(xPos, 0.0, 25.0));
    }
    
    Ptr<UniformRandomVariable> rv = CreateObject<UniformRandomVariable>();
    rv->SetStream(12345);
    
    for (uint32_t globalUeId = 0; globalUeId < totalUes; globalUeId++) {
        GlobalUeInfo ueInfo;
        ueInfo.globalUeId = globalUeId;
        
        ueInfo.sliceType = globalUeId % 3;  // 0=URLLC, 1=eMBB, 2=mMTC
        
        uint32_t baseGnbIndex = rv->GetInteger(0, gNbNum - 1);
        double distance = rv->GetValue(0, 50.0);
        double angle = rv->GetValue(0, 2 * M_PI);
        
        double x = globalGnbPositions[baseGnbIndex].x + distance * cos(angle);
        double y = globalGnbPositions[baseGnbIndex].y + distance * sin(angle);
        double z = 1.5;  // UE height
        
        ueInfo.position = Vector(x, y, z);
        
        double minDistance = std::numeric_limits<double>::max();
        uint32_t closestGnbIndex = 0;
        
        for (uint32_t j = 0; j < globalGnbPositions.size(); ++j) {
            double d = std::sqrt(
                std::pow(x - globalGnbPositions[j].x, 2) +
                std::pow(y - globalGnbPositions[j].y, 2)
            );
            
            if (d < minDistance) {
                minDistance = d;
                closestGnbIndex = j;
            }
        }
        
        ueInfo.servingGnbId = closestGnbIndex;
        ueInfo.distanceToGnb = minDistance;
        
        allUeInfos.push_back(ueInfo);
    }
    
    for (auto& ueInfo : allUeInfos) {
        globalUeIdToInfoMap[ueInfo.globalUeId] = &ueInfo;
    }
    
    NS_LOG_INFO("Initialized " << allUeInfos.size() << " UEs with global IDs and positions");
}

std::vector<GlobalUeInfo*> GetUesForBatch(uint32_t batchNumber) {
    std::vector<GlobalUeInfo*> batchUes;
    
    uint32_t startIdx = batchNumber * batchSize;
    uint32_t endIdx = std::min(startIdx + batchSize, totalUes);
    
    for (uint32_t i = startIdx; i < endIdx; i++) {
        if (i < allUeInfos.size()) {
            batchUes.push_back(&allUeInfos[i]);
        }
    }
    
    NS_LOG_INFO("Batch " << batchNumber << ": UEs " << startIdx << " to " << (endIdx-1) 
               << " (total " << batchUes.size() << " UEs)");
    return batchUes;
}

void SetupImsiMapping(NodeContainer ueNodes, NetDeviceContainer ueDevices) {
    NS_LOG_INFO("Setting up IMSI mapping...");
    
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        Ptr<Node> ueNode = ueNodes.Get(i);
        Ptr<NetDevice> ueDevice = ueDevices.Get(i);
        uint32_t nodeId = ueNode->GetId();
        
        ueSinrData[nodeId] = UeSinrData();
        uePowerData[nodeId] = UePowerData();
        
        uint64_t imsi = 0;
        Ptr<NrUeNetDevice> nrUeDevice = DynamicCast<NrUeNetDevice>(ueDevice);
        if (nrUeDevice) {
            imsi = nrUeDevice->GetImsi();
        } else {
            imsi = nodeId + 1;
        }
        
        if (imsi > 0) {
            imsiToNodeIdMap[imsi] = nodeId;
        }
    }
}

void SetupSinrTracing(NodeContainer ueNodes, NetDeviceContainer ueDevices, Ptr<NrHelper> nrHelper) {
    NS_LOG_INFO("Setting up SINR tracing...");
    sinrSamplingRandom = CreateObject<UniformRandomVariable>();
    sinrSamplingRandom->SetAttribute("Min", DoubleValue(0.0));
    sinrSamplingRandom->SetAttribute("Max", DoubleValue(1.0));
    sinrSamplingRandom->SetStream(12345 + currentIteration * 1000);
    
    int connectedCount = 0;    
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        Ptr<Node> ueNode = ueNodes.Get(i);
        Ptr<NetDevice> ueDevice = ueDevices.Get(i);
        
        Ptr<NrUePhy> uePhy = nrHelper->GetUePhy(ueDevice, 0);
        if (!uePhy) continue;
        
        try {
            bool connected = uePhy->TraceConnectWithoutContext(
                "DlSinr", 
                MakeCallback(&OptimizedSinrCallback)
            );
            
            if (connected) {
                connectedCount++;
                continue;
            }
        } catch (...) {}
  
        try {
            bool connected = uePhy->TraceConnectWithoutContext(
                "UlSinr", 
                MakeCallback(&OptimizedSinrCallback)
            );
            if (connected) {
                connectedCount++;
            }
        } catch (...) {}
    }
    
    NS_LOG_INFO("Connected SINR tracing for " << connectedCount << " UEs");
}

void SetupPowerTracing(NodeContainer ueNodes, NetDeviceContainer ueDevices) {
    NS_LOG_INFO("Setting up power consumption tracing...");
    
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        Ptr<Node> ueNode = ueNodes.Get(i);
        uint32_t nodeId = ueNode->GetId();
        auto& powerInfo = uePowerData[nodeId];        
        auto it = nodeIdToUeInfoMap.find(nodeId);
        if (it != nodeIdToUeInfoMap.end()) {
            GlobalUeInfo* ueInfo = it->second;
            double distance = ueInfo->distanceToGnb;
            
            Ptr<UniformRandomVariable> rand = CreateObject<UniformRandomVariable>();
            rand->SetStream(ueInfo->globalUeId + 1000 + currentIteration * 10000);
            
            double distanceBasedTxPower = 13.0;
            if (distance > 10.0) {
                distanceBasedTxPower += 0.1 * (distance - 10.0);
                if (distanceBasedTxPower > 23.0) distanceBasedTxPower = 23.0;
            }
            distanceBasedTxPower += rand->GetValue(-2, 2);
            
            double distanceBasedRxPower = -60.0 - 20 * log10(distance/10.0);
            if (distanceBasedRxPower < -100.0) distanceBasedRxPower = -100.0;
            distanceBasedRxPower += rand->GetValue(-5, 5);
            
            powerInfo.AddPowerSample(distanceBasedTxPower, distanceBasedRxPower);
            powerInfo.lastUpdateTime = Seconds(0);
            
            NS_LOG_DEBUG("Power for UE " << ueInfo->globalUeId << " (node " << nodeId 
                         << "): TX=" << distanceBasedTxPower << " dBm, RX=" 
                         << distanceBasedRxPower << " dBm, distance=" << distance << " m");
        } else {
            // Fallback initialization
            Ptr<UniformRandomVariable> rand = CreateObject<UniformRandomVariable>();
            rand->SetStream(nodeId + 1000 + currentIteration * 10000);
            double initialTxPower = 13.0 + rand->GetValue(0, 10);
            double initialRxPower = -75.0 - rand->GetValue(0, 15);
            
            powerInfo.AddPowerSample(initialTxPower, initialRxPower);
            powerInfo.lastUpdateTime = Seconds(0);
        }
    }
}

void PeriodicPowerSampling() {
    Time samplingInterval = MilliSeconds(100);
    
    for (auto& pair : uePowerData) {
        uint32_t nodeId = pair.first;
        auto& powerInfo = pair.second;
        
        Time now = Simulator::Now();
        
        if (powerInfo.lastUpdateTime.GetSeconds() > 0) {
            double duration = (now - powerInfo.lastUpdateTime).GetSeconds();
            double avgTxPowerW = pow(10, (powerInfo.GetAverageTxPower() - 30) / 10);
            double avgRxPowerW = pow(10, (powerInfo.GetAverageRxPower() - 30) / 10);
            
            powerInfo.totalEnergyJoules += (avgTxPowerW + avgRxPowerW) * duration;
        }
        powerInfo.lastUpdateTime = now;
        
        auto it = nodeIdToUeInfoMap.find(nodeId);
        if (it != nodeIdToUeInfoMap.end()) {
            GlobalUeInfo* ueInfo = it->second;
            double distance = ueInfo->distanceToGnb;
            
            Ptr<UniformRandomVariable> rand = CreateObject<UniformRandomVariable>();
            rand->SetStream(ueInfo->globalUeId + 2000 + currentIteration * 10000);
            
            double distanceBasedTxPower = 13.0;
            if (distance > 10.0) {
                distanceBasedTxPower += 0.1 * (distance - 10.0);
                if (distanceBasedTxPower > 23.0) distanceBasedTxPower = 23.0;
            }
            double txPowerDbm = distanceBasedTxPower + rand->GetValue(-2, 2);
            
            double distanceBasedRxPower = -60.0 - 20 * log10(distance/10.0);
            if (distanceBasedRxPower < -100.0) distanceBasedRxPower = -100.0;
            double rxPowerDbm = distanceBasedRxPower + rand->GetValue(-5, 5);
            
            powerInfo.AddPowerSample(txPowerDbm, rxPowerDbm);
        } else {
            Ptr<UniformRandomVariable> rand = CreateObject<UniformRandomVariable>();
            rand->SetStream(nodeId + 2000 + currentIteration * 10000);
            
            double txPowerDbm = 13.0 + rand->GetValue(-2, 8);
            double rxPowerDbm = -80.0 + rand->GetValue(-10, 10);
            
            powerInfo.AddPowerSample(txPowerDbm, rxPowerDbm);
        }
    }
    Simulator::Schedule(samplingInterval, &PeriodicPowerSampling);
}

void CalculateDistanceAndApproximateSinr(NodeContainer ueNodes, double frequency, double txPower, double bandwidthBand1, double bandwidthBand2, double bandwidthBand3) {
    NS_LOG_INFO("Calculating UE distances and approximate SINR...");
    
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        Ptr<Node> ueNode = ueNodes.Get(i);
        uint32_t nodeId = ueNode->GetId();        
        auto it = nodeIdToUeInfoMap.find(nodeId);
        
        if (it != nodeIdToUeInfoMap.end()) {
            GlobalUeInfo* ueInfo = it->second;
            double minDistance = ueInfo->distanceToGnb;
            
            auto sinrIt = ueSinrData.find(nodeId);                        
            if (sinrIt == ueSinrData.end() || sinrIt->second.count == 0) {
                if (minDistance < 1.0) minDistance = 1.0;
                double bandwidthHz;
                if (ueInfo->sliceType == 0) {      
                    bandwidthHz = bandwidthBand1;
                }
                else if (ueInfo->sliceType == 1) {      
                    bandwidthHz = bandwidthBand2;
                }
                else if (ueInfo->sliceType == 2) {     
                    bandwidthHz = bandwidthBand3;
                }
                else {                    
                    NS_FATAL_ERROR("Unsupported sliceType=" << ueInfo->sliceType
                       << " for UE " << ueInfo->globalUeId
                       << " (nodeId=" << nodeId << "). "
                       << "Must be 0=URLLC, 1=eMBB, 2=mMTC.");
                }
                // double fsplDb = 20 * log10(minDistance) + 20 * log10(frequency) - 147.55 + 20.0; // Friis for Sub-6
                double fsplDb = 20.0 * log10(minDistance) + 20.0 * log10(frequency) - 147.55;  // Friis for mmWave
                double mmwExtraLoss = 8.0;       
                double pathLossDb = fsplDb + mmwExtraLoss;
                double rxPowerDbm = txPower - pathLossDb;
                double noiseFloorDbm = -174.0 + 10.0 * log10(bandwidthHz);
                double noiseFigureDb = 9.0;
                double noisePower = noiseFloorDbm + noiseFigureDb;
                double sinrDb = rxPowerDbm - noisePower;
                
                if (ueInfo->sliceType == 0)      sinrDb += 1.5;
                else if (ueInfo->sliceType == 2) sinrDb -= 1.0;

                if (sinrDb > 40.0) sinrDb = 40.0;
                if (sinrDb < -20.0) sinrDb = -20.0;

                ueSinrData[nodeId].AddSample(sinrDb);
            }
        }
    }
}

void PrintSinrStats() {
    NS_LOG_UNCOND("\n=== SINR STATISTICS ===");
    
    int uesWithMeasurements = 0;
    double totalSinr = 0.0;
    
    for (const auto& pair : ueSinrData) {
        uint32_t nodeId = pair.first;
        const UeSinrData& sinrInfo = pair.second;
        
        if (sinrInfo.count > 0) {
            uesWithMeasurements++;
            double avgSinr = sinrInfo.GetAverage();
            totalSinr += avgSinr;            
            uint32_t globalUeId = 0;
            auto it = nodeIdToUeInfoMap.find(nodeId);
            if (it != nodeIdToUeInfoMap.end()) {
                globalUeId = it->second->globalUeId;
            }
            
            NS_LOG_UNCOND("UE " << globalUeId << " (node " << nodeId << "): " 
                         << sinrInfo.count << " samples, Avg SINR: " << avgSinr << " dB");
        }
    }
    
    if (uesWithMeasurements > 0) {
        NS_LOG_UNCOND("Average SINR across all UEs: " << totalSinr / uesWithMeasurements << " dB");
    }
}

void PrintPowerStats() {
    NS_LOG_UNCOND("\n=== POWER CONSUMPTION STATISTICS ===");
    
    double totalEnergy = 0.0;
    double totalAvgTxPower = 0.0;
    double totalAvgRxPower = 0.0;
    int ueCount = 0;
    
    for (const auto& pair : uePowerData) {
        uint32_t nodeId = pair.first;
        const UePowerData& powerInfo = pair.second;
        
        if (powerInfo.txPowerSamples.size() > 0) {
            ueCount++;
            double avgTxPower = powerInfo.GetAverageTxPower();
            double avgRxPower = powerInfo.GetAverageRxPower();
            double energy = powerInfo.totalEnergyJoules;
            
            totalEnergy += energy;
            totalAvgTxPower += avgTxPower;
            totalAvgRxPower += avgRxPower; 
            
            uint32_t globalUeId = 0;
            auto it = nodeIdToUeInfoMap.find(nodeId);
            if (it != nodeIdToUeInfoMap.end()) {
                globalUeId = it->second->globalUeId;
            }
            
            NS_LOG_UNCOND("UE " << globalUeId << " (node " << nodeId << "): Avg TX=" 
                         << avgTxPower << " dBm, Avg RX=" << avgRxPower << " dBm, "
                         << "Total Energy=" << energy << " J");
        }
    }
    
    if (ueCount > 0) {
        NS_LOG_UNCOND("\nSummary across all UEs:");
        NS_LOG_UNCOND("Average TX Power: " << totalAvgTxPower / ueCount << " dBm");
        NS_LOG_UNCOND("Average RX Power: " << totalAvgRxPower / ueCount << " dBm");
        NS_LOG_UNCOND("Total Energy Consumption: " << totalEnergy << " J");
        NS_LOG_UNCOND("Average Energy per UE: " << totalEnergy / ueCount << " J");
        
        if (totalEnergy > 0) {           
            NS_LOG_UNCOND("Energy Efficiency metric available in CSV output");
        }
    }
}

void CreateResultsDirectory() {
    if (!std::filesystem::exists(resultsDir)) {
        std::filesystem::create_directory(resultsDir);
        NS_LOG_INFO("Created results directory: " << resultsDir);
    }
}

bool CheckBatchFileComplete(uint32_t batchNumber) {
    std::string filename = resultsDir + "batch_" + std::to_string(batchNumber) + ".csv";
    
    if (!std::filesystem::exists(filename)) {
        NS_LOG_INFO("Batch file " << filename << " does not exist.");
        return false;
    }
    
    NS_LOG_INFO("Checking completeness of batch file: " << filename);
    
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        NS_LOG_WARN("Could not open file for reading: " << filename);
        return false;
    }
    
    std::string line;
    uint32_t lineCount = 0;
    
    while (std::getline(inFile, line)) {
        if (!line.empty() && lineCount == 0) {
            lineCount++;
            continue;
        }
        if (!line.empty()) {
            lineCount++;
        }
    }
    
    inFile.close();
    
    uint32_t expectedMinRecords = totalIterations * batchSize;
    
    NS_LOG_INFO("File has " << lineCount << " data records (excluding header)");
    NS_LOG_INFO("Expected minimum records: " << expectedMinRecords);
    
    return (lineCount >= expectedMinRecords);
}

uint32_t FindStartingIteration(uint32_t batchNumber) {
    std::string filename = resultsDir + "batch_" + std::to_string(batchNumber) + ".csv";
    
    if (!std::filesystem::exists(filename)) {
        return 0; 
    }
    
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        return 0;
    }
    
    std::string line;
    uint32_t maxIterationFound = 0;
    
    std::getline(inFile, line);
    
    while (std::getline(inFile, line)) {
        if (!line.empty()) {
            size_t commaPos = line.find(',');
            if (commaPos != std::string::npos) {
                std::string iterationStr = line.substr(0, commaPos);
                try {
                    uint32_t iteration = std::stoi(iterationStr);
                    if (iteration > maxIterationFound) {
                        maxIterationFound = iteration;
                    }
                } catch (...) {
                    // Ignore conversion errors
                }
            }
        }
    }
    
    inFile.close();
    
    uint32_t startIteration = maxIterationFound + 1;
    
    NS_LOG_INFO("Found max iteration in file: " << maxIterationFound);
    NS_LOG_INFO("Will start from iteration: " << startIteration);
    
    return startIteration;
}

void OpenBatchOutputFile() {
    std::string filename = resultsDir + "batch_" + std::to_string(currentBatch) + ".csv";
    
    bool needHeader = true;
    bool fileExists = std::filesystem::exists(filename);
    
    if (fileExists && !std::filesystem::is_empty(filename)) {
        std::ifstream inFile(filename);
        std::string firstLine;
        if (std::getline(inFile, firstLine)) {
            if (firstLine.find("Iteration") != std::string::npos && 
                firstLine.find("GlobalUEId") != std::string::npos) {
                needHeader = false;
                NS_LOG_INFO("File already has proper header: " << filename);
            } else {
                NS_LOG_WARN("File exists but first line doesn't look like a header!");
                NS_LOG_WARN("First line: " << firstLine);
            }
        }
        inFile.close();
    }
    
    std::ofstream outFile;
    
    if (needHeader) {
        if (fileExists && !std::filesystem::is_empty(filename)) {
            NS_LOG_INFO("File exists without header, recreating with header: " << filename);
            
            std::ifstream inFile(filename);
            std::stringstream buffer;
            buffer << inFile.rdbuf(); 
            inFile.close();
            
            outFile.open(filename, std::ios::out);
            outFile << "Iteration,GlobalUEId,FlowID,SourceIP,DestinationIP,SourcePort,DestinationPort,Protocol,"
                    << "TxPackets,RxPackets,TxBytes,RxBytes,Throughput(Mbps),"
                    << "AvgDelay(s),AvgJitter(s),PacketLossRatio(%),SliceType,Numerology,SliceBW(MHz),Direction,"
                    << "SINR_dB,Distance_m,ServingGnbId,AvgTxPower_dBm,AvgRxPower_dBm,TotalEnergy_J\n";
            outFile << buffer.str(); 
        } else {
            outFile.open(filename, std::ios::out);
            outFile << "Iteration,GlobalUEId,FlowID,SourceIP,DestinationIP,SourcePort,DestinationPort,Protocol,"
                    << "TxPackets,RxPackets,TxBytes,RxBytes,Throughput(Mbps),"
                    << "AvgDelay(s),AvgJitter(s),PacketLossRatio(%),SliceType,Numerology,SliceBW(MHz),Direction,"
                    << "SINR_dB,Distance_m,ServingGnbId,AvgTxPower_dBm,AvgRxPower_dBm,TotalEnergy_J\n";
            NS_LOG_INFO("Created new file with header: " << filename);
        }
    } else {
        outFile.open(filename, std::ios::app);
        NS_LOG_INFO("Opened existing file in append mode: " << filename);
    }
    
    outputFiles.push_back(std::move(outFile));
    NS_LOG_INFO("Opened output file: " << filename << " (mode: " << (fileExists ? "append" : "new") << ")");
}

void CloseOutputFiles() {
    for (auto& file : outputFiles) {
        if (file.is_open()) {
            file.close();
        }
    }
    outputFiles.clear();
}

void RunBatchSimulation(uint32_t batchNumber, uint32_t iteration, const std::vector<GlobalUeInfo*>& batchUes) {
    NS_LOG_UNCOND("\n=========================================");        
    NS_LOG_UNCOND("Total UEs in batch: " << batchUes.size()); 
    NS_LOG_UNCOND("Current Seed: " << seeds[iteration]);
    NS_LOG_UNCOND("=========================================");
    
    RngSeedManager::SetSeed(seeds[iteration]);
    RngSeedManager::SetRun(iteration + 1);
    
    uint16_t gNbNum = 2;
    bool logging = false;
    bool enableOfdma = false;
    
    bool enableSinrTracing = true;
    double sinrSampleRate = 0.05;

    uint32_t udpPacketSizeULL = 32;
    uint32_t udpPacketSizeEMBB = 1400;
    uint32_t udpPacketSizeMMTC = 32;
    
    double lambdaULL = 1000.0;
    double lambdaEMBB = 500.0;
    double lambdaMMTC = 1.0;

    Time simTime = Seconds(3.0);
    Time udpAppStartTime = Seconds(0.5);

    uint16_t numerologyBwp1 = 4;
    double centralFrequencyBand1 = 28.0e9;
    double bandwidthBand1 = 36.72e6;
    
    uint16_t numerologyBwp2 = 3;
    double centralFrequencyBand2 = 28.0e9;
    double bandwidthBand2 = 54.36e6;
    
    uint16_t numerologyBwp3 = 2;
    double centralFrequencyBand3 = 28.0e9;
    double bandwidthBand3 = 7.2e6;
    
    double totalTxPower = 35.0;
    double ueTxPower = 23.0;

    std::string simTag = "nr-2gnb-" + std::to_string(batchUes.size()) + "ue-3slice-ul";
    std::string outputDir = "net_sim_res_batch/";

    enableSinrTracingConfig = enableSinrTracing;
    sinrSamplingRate = sinrSampleRate;

    NS_ABORT_IF(centralFrequencyBand1 < 0.5e9 || centralFrequencyBand1 > 100e9);
    NS_ABORT_IF(centralFrequencyBand2 < 0.5e9 || centralFrequencyBand2 > 100e9);
    NS_ABORT_IF(centralFrequencyBand3 < 0.5e9 || centralFrequencyBand3 > 100e9);

    if (logging) {
        LogComponentEnable("UdpClient", LOG_LEVEL_WARN);
        LogComponentEnable("UdpServer", LOG_LEVEL_WARN);
        LogComponentEnable("NrPdcp", LOG_LEVEL_WARN);
        LogComponentEnable("CttcNrDemo", LOG_LEVEL_WARN);
    }
    
    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(999999999));
    Config::SetDefault("ns3::NrRlcAm::MaxTxBufferSize", UintegerValue(999999999));
    
    // QoS for uplink configs
    Config::SetDefault("ns3::NrUePhy::EnableUplinkPowerControl", BooleanValue(true));
    Config::SetDefault("ns3::NrUePowerControl::ClosedLoop", BooleanValue(true));
    Config::SetDefault("ns3::NrUePowerControl::AccumulationEnabled", BooleanValue(true));
         
    int64_t randomStream = 1;
    
    NodeContainer gnbNodes;
    NodeContainer ueNodes;
    gnbNodes.Create(gNbNum);    
    ueNodes.Create(batchUes.size());  
    
    nodeIdToUeInfoMap.clear();
    
    for (uint32_t i = 0; i < batchUes.size(); ++i) {
        GlobalUeInfo* ueInfo = batchUes[i];
        Ptr<Node> ueNode = ueNodes.Get(i);
        ueInfo->assignedNodeId = ueNode->GetId();
        nodeIdToUeInfoMap[ueNode->GetId()] = ueInfo;
        
        NS_LOG_INFO("Global UE " << ueInfo->globalUeId << " assigned to node " 
                   << ueNode->GetId() << " at position (" 
                   << ueInfo->position.x << ", " << ueInfo->position.y << ", " 
                   << ueInfo->position.z << ")");
    }
    
    Ptr<ListPositionAllocator> gnbPositionAlloc = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < gNbNum; ++i) {
        gnbPositionAlloc->Add(globalGnbPositions[i]);
        NS_LOG_INFO("gNB " << i << " at position: (" 
                    << globalGnbPositions[i].x << ", " 
                    << globalGnbPositions[i].y << ", " 
                    << globalGnbPositions[i].z << ")");
    }
    
    MobilityHelper gnbMobility;
    gnbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    gnbMobility.SetPositionAllocator(gnbPositionAlloc);
    gnbMobility.Install(gnbNodes);
    
    MobilityHelper ueMobility;
    ueMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    
    Ptr<ListPositionAllocator> uePositionAlloc = CreateObject<ListPositionAllocator>();
    for (GlobalUeInfo* ueInfo : batchUes) {
        uePositionAlloc->Add(ueInfo->position);
    }
    ueMobility.SetPositionAllocator(uePositionAlloc);
    ueMobility.Install(ueNodes);
    
    NodeContainer ueUrlLcContainer;
    NodeContainer ueEmbbContainer;
    NodeContainer ueMmtcContainer;
    
    for (uint32_t i = 0; i < batchUes.size(); ++i) {
        GlobalUeInfo* ueInfo = batchUes[i];
        Ptr<Node> ueNode = ueNodes.Get(i);
        
        switch (ueInfo->sliceType) {
            case 0: // URLLC
                ueUrlLcContainer.Add(ueNode);
                break;
            case 1: // eMBB
                ueEmbbContainer.Add(ueNode);
                break;
            case 2: // mMTC
                ueMmtcContainer.Add(ueNode);
                break;
        }
    }
    
    NS_LOG_INFO("Created " << ueNodes.GetN() << " user terminals and " 
                           << gnbNodes.GetN() << " gNBs");
    NS_LOG_INFO("URLLC UEs: " << ueUrlLcContainer.GetN());
    NS_LOG_INFO("eMBB UEs: " << ueEmbbContainer.GetN());
    NS_LOG_INFO("mMTC UEs: " << ueMmtcContainer.GetN());

    Ptr<NrPointToPointEpcHelper> nrEpcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> idealBeamformingHelper = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();

    nrHelper->SetBeamformingHelper(idealBeamformingHelper);
    nrHelper->SetEpcHelper(nrEpcHelper);

    Config::SetDefault("ns3::ThreeGppChannelModel::UpdatePeriod", TimeValue(MilliSeconds(0)));
    nrHelper->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(0)));
    nrHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(false));

    idealBeamformingHelper->SetAttribute("BeamformingMethod",
                                         TypeIdValue(DirectPathBeamforming::GetTypeId()));

    nrEpcHelper->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));

    std::string subType = !enableOfdma ? "Tdma" : "Ofdma";
    std::string sched = "Qos";       
    std::string schedulerType = "ns3::NrMacScheduler" + subType + sched;
    NS_LOG_INFO("SchedulerType: " << schedulerType);
    nrHelper->SetSchedulerTypeId(TypeId::LookupByName(schedulerType));
   
    uint16_t mcsTable = 2;
    std::string errorModel = "ns3::NrEesmIrT" + std::to_string(mcsTable);
    nrHelper->SetDlErrorModel(errorModel);
    nrHelper->SetUlErrorModel(errorModel);
    
    nrHelper->SetGnbDlAmcAttribute("AmcModel", EnumValue(NrAmc::ErrorModel));
    nrHelper->SetGnbUlAmcAttribute("AmcModel", EnumValue(NrAmc::ErrorModel));
    
    nrHelper->SetSchedulerAttribute("FixedMcsDl", BooleanValue(false));
    nrHelper->SetSchedulerAttribute("FixedMcsUl", BooleanValue(false));

    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(2));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(4));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
                                    PointerValue(CreateObject<IsotropicAntennaModel>()));

    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(4));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(8));
    nrHelper->SetGnbAntennaAttribute("AntennaElement",
                                     PointerValue(CreateObject<IsotropicAntennaModel>()));

    BandwidthPartInfoPtrVector allBwps;
    CcBwpCreator ccBwpCreator;
    const uint8_t numCcPerBand = 1;

    CcBwpCreator::SimpleOperationBandConf bandConf1(centralFrequencyBand1,
                                                    bandwidthBand1,
                                                    numCcPerBand,
                                                    BandwidthPartInfo::UMi_StreetCanyon);
    bandConf1.m_numBwp = 1;

    CcBwpCreator::SimpleOperationBandConf bandConf2(centralFrequencyBand2,
                                                    bandwidthBand2,
                                                    numCcPerBand,
                                                    BandwidthPartInfo::UMi_StreetCanyon);
    bandConf2.m_numBwp = 1;

    CcBwpCreator::SimpleOperationBandConf bandConf3(centralFrequencyBand3,
                                                    bandwidthBand3,
                                                    numCcPerBand,
                                                    BandwidthPartInfo::UMi_StreetCanyon);
    bandConf3.m_numBwp = 1;

    OperationBandInfo band1 = ccBwpCreator.CreateOperationBandContiguousCc(bandConf1);
    OperationBandInfo band2 = ccBwpCreator.CreateOperationBandContiguousCc(bandConf2);
    OperationBandInfo band3 = ccBwpCreator.CreateOperationBandContiguousCc(bandConf3);

    nrHelper->InitializeOperationBand(&band1);
    nrHelper->InitializeOperationBand(&band2);
    nrHelper->InitializeOperationBand(&band3);
    
    allBwps = CcBwpCreator::GetAllBwps({band1, band2, band3});

    uint32_t bwpIdUrlLc = 0;
    uint32_t bwpIdEmbb = 1;
    uint32_t bwpIdMmtc = 2;

    nrHelper->SetGnbBwpManagerAlgorithmAttribute("GBR_CONV_VOICE", UintegerValue(bwpIdUrlLc));
    nrHelper->SetGnbBwpManagerAlgorithmAttribute("NGBR_VIDEO_TCP_PREMIUM", UintegerValue(bwpIdEmbb));
    nrHelper->SetGnbBwpManagerAlgorithmAttribute("NGBR_VIDEO_TCP_DEFAULT", UintegerValue(bwpIdMmtc));
    
    nrHelper->SetUeBwpManagerAlgorithmAttribute("GBR_CONV_VOICE", UintegerValue(bwpIdUrlLc));
    nrHelper->SetUeBwpManagerAlgorithmAttribute("NGBR_VIDEO_TCP_PREMIUM", UintegerValue(bwpIdEmbb));
    nrHelper->SetUeBwpManagerAlgorithmAttribute("NGBR_VIDEO_TCP_DEFAULT", UintegerValue(bwpIdMmtc));
  
    NetDeviceContainer gnbNetDev = nrHelper->InstallGnbDevice(gnbNodes, allBwps);
    NetDeviceContainer ueUrlLcNetDev = nrHelper->InstallUeDevice(ueUrlLcContainer, allBwps);
    NetDeviceContainer ueEmbbNetDev = nrHelper->InstallUeDevice(ueEmbbContainer, allBwps);
    NetDeviceContainer ueMmtcNetDev = nrHelper->InstallUeDevice(ueMmtcContainer, allBwps);

    NetDeviceContainer ueNetDev;
    ueNetDev.Add(ueUrlLcNetDev);
    ueNetDev.Add(ueEmbbNetDev);
    ueNetDev.Add(ueMmtcNetDev);

    randomStream += nrHelper->AssignStreams(gnbNetDev, randomStream);
    randomStream += nrHelper->AssignStreams(ueUrlLcNetDev, randomStream);
    randomStream += nrHelper->AssignStreams(ueEmbbNetDev, randomStream);
    randomStream += nrHelper->AssignStreams(ueMmtcNetDev, randomStream);

    ueSinrData.clear();
    uePowerData.clear();  
    imsiToNodeIdMap.clear();
    ipToUeInfoMap.clear(); 

    double x = pow(10, totalTxPower / 10);
    double totalBandwidth = bandwidthBand1 + bandwidthBand2 + bandwidthBand3;

    for (uint32_t i = 0; i < gnbNetDev.GetN(); ++i) {
        nrHelper->GetGnbPhy(gnbNetDev.Get(i), 0)->SetAttribute("Numerology", UintegerValue(numerologyBwp1));
        double txPower = 10 * log10((bandwidthBand1 / totalBandwidth) * x);
        nrHelper->GetGnbPhy(gnbNetDev.Get(i), 0)->SetAttribute("TxPower", DoubleValue(txPower));
    }

    for (uint32_t i = 0; i < gnbNetDev.GetN(); ++i) {
        nrHelper->GetGnbPhy(gnbNetDev.Get(i), 1)->SetAttribute("Numerology", UintegerValue(numerologyBwp2));
        double txPower = 10 * log10((bandwidthBand2 / totalBandwidth) * x);
        nrHelper->GetGnbPhy(gnbNetDev.Get(i), 1)->SetAttribute("TxPower", DoubleValue(txPower));
    }

    for (uint32_t i = 0; i < gnbNetDev.GetN(); ++i) {
        nrHelper->GetGnbPhy(gnbNetDev.Get(i), 2)->SetAttribute("Numerology", UintegerValue(numerologyBwp3));
        double txPower = 10 * log10((bandwidthBand3 / totalBandwidth) * x);
        nrHelper->GetGnbPhy(gnbNetDev.Get(i), 2)->SetAttribute("TxPower", DoubleValue(txPower));
    }

    nrHelper->SetUePhyAttribute("TxPower", DoubleValue(ueTxPower));

    nrHelper->UpdateDeviceConfigs(gnbNetDev);
    nrHelper->UpdateDeviceConfigs(ueUrlLcNetDev);
    nrHelper->UpdateDeviceConfigs(ueEmbbNetDev);
    nrHelper->UpdateDeviceConfigs(ueMmtcNetDev);

    SetupImsiMapping(ueNodes, ueNetDev);
    
    // Set up SINR tracing
    if (enableSinrTracingConfig) {
        SetupSinrTracing(ueNodes, ueNetDev, nrHelper);
    }
    
    SetupPowerTracing(ueNodes, ueNetDev);
    
    Simulator::Schedule(MilliSeconds(100), &PeriodicPowerSampling);
    
    CalculateDistanceAndApproximateSinr(ueNodes, centralFrequencyBand1, ueTxPower, bandwidthBand1, bandwidthBand2, bandwidthBand3);

    NS_LOG_UNCOND("\n=== INITIAL UE INFORMATION ===");
    
    std::map<uint32_t, uint32_t> ueCountPerGnb;
    
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        Ptr<Node> ueNode = ueNodes.Get(i);
        uint32_t nodeId = ueNode->GetId();
        
        auto it = nodeIdToUeInfoMap.find(nodeId);
        if (it != nodeIdToUeInfoMap.end()) {
            GlobalUeInfo* ueInfo = it->second;
            
            double sinrDb = GetLatestSinr(nodeId);
            double avgTxPower = GetAverageTxPower(nodeId);
            double avgRxPower = GetAverageRxPower(nodeId);
            
            ueCountPerGnb[ueInfo->servingGnbId]++;
            
            NS_LOG_UNCOND("  UE " << ueInfo->globalUeId << " --> node " << nodeId 
                        << ": Distance=" << ueInfo->distanceToGnb 
                        << " m, Serving gNB=" << ueInfo->servingGnbId 
                        << ", SINR=" << sinrDb << " dB"
                        << ", Avg TX=" << avgTxPower << " dBm"
                        << ", Avg RX=" << avgRxPower << " dBm"
                        << ", Slice=" << (ueInfo->sliceType == 0 ? "URLLC" : 
                                          ueInfo->sliceType == 1 ? "eMBB" : "mMTC"));
        }
    }
    
    NS_LOG_UNCOND("\n=== UE DISTRIBUTION PER GNB ===");
    uint32_t totalConnectedUes = 0;
    for (const auto& entry : ueCountPerGnb) {
        NS_LOG_UNCOND("  gNB " << entry.first << ": " << entry.second << " UEs");
        totalConnectedUes += entry.second;
    }
    NS_LOG_UNCOND("  Total connected UEs: " << totalConnectedUes << "/" << ueNodes.GetN());
    NS_LOG_UNCOND("\n=== UPLINK NETWORK SLICING CONFIGURATION ===");
    NS_LOG_UNCOND("  Total Bandwidth: " << totalBandwidth/1e6 << " MHz");
    NS_LOG_UNCOND("  URLLC Slice: " << bandwidthBand1/1e6 << " MHz at " << centralFrequencyBand1/1e9 << " GHz (μ=" << numerologyBwp1 << ")");
    NS_LOG_UNCOND("  eMBB Slice:  " << bandwidthBand2/1e6 << " MHz at " << centralFrequencyBand2/1e9 << " GHz (μ=" << numerologyBwp2 << ")");
    NS_LOG_UNCOND("  mMTC Slice:  " << bandwidthBand3/1e6 << " MHz at " << centralFrequencyBand3/1e9 << " GHz (μ=" << numerologyBwp3 << ")");
    
    NS_LOG_UNCOND("\nRunning simulation Batch " << batchNumber << " (Iteration " << iteration << ", Seed " << seeds[iteration] << ")... ");
    
    Ptr<Node> pgw = nrEpcHelper->GetPgwNode();
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);
    InternetStackHelper internet;
    internet.Install(remoteHostContainer);
    internet.Install(ueNodes);

    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(2500));
    p2ph.SetChannelAttribute("Delay", TimeValue(Seconds(0.000)));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);
    
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);
    
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting = 
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    Ipv4InterfaceContainer ueUrlLcIpIface = nrEpcHelper->AssignUeIpv4Address(ueUrlLcNetDev);
    Ipv4InterfaceContainer ueEmbbIpIface = nrEpcHelper->AssignUeIpv4Address(ueEmbbNetDev);
    Ipv4InterfaceContainer ueMmtcIpIface = nrEpcHelper->AssignUeIpv4Address(ueMmtcNetDev);

    for (uint32_t i = 0; i < ueUrlLcContainer.GetN(); ++i) {
        Ptr<Node> ueNode = ueUrlLcContainer.Get(i);
        Ipv4Address ipAddr = ueUrlLcIpIface.GetAddress(i);
        auto it = nodeIdToUeInfoMap.find(ueNode->GetId());
        if (it != nodeIdToUeInfoMap.end()) {
            it->second->ipAddress = ipAddr;
            ipToUeInfoMap[ipAddr] = it->second;
            NS_LOG_INFO("URLLC UE " << it->second->globalUeId << " assigned IP: " << ipAddr);
        }
    }
    
    for (uint32_t i = 0; i < ueEmbbContainer.GetN(); ++i) {
        Ptr<Node> ueNode = ueEmbbContainer.Get(i);
        Ipv4Address ipAddr = ueEmbbIpIface.GetAddress(i);
        auto it = nodeIdToUeInfoMap.find(ueNode->GetId());
        if (it != nodeIdToUeInfoMap.end()) {
            it->second->ipAddress = ipAddr;
            ipToUeInfoMap[ipAddr] = it->second;
            NS_LOG_INFO("eMBB UE " << it->second->globalUeId << " assigned IP: " << ipAddr);
        }
    }
    
    for (uint32_t i = 0; i < ueMmtcContainer.GetN(); ++i) {
        Ptr<Node> ueNode = ueMmtcContainer.Get(i);
        Ipv4Address ipAddr = ueMmtcIpIface.GetAddress(i);
        auto it = nodeIdToUeInfoMap.find(ueNode->GetId());
        if (it != nodeIdToUeInfoMap.end()) {
            it->second->ipAddress = ipAddr;
            ipToUeInfoMap[ipAddr] = it->second;
            NS_LOG_INFO("mMTC UE " << it->second->globalUeId << " assigned IP: " << ipAddr);
        }
    }

    Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress(1);

    for (uint32_t j = 0; j < ueNodes.GetN(); ++j) {
        Ptr<Node> ueNode = ueNodes.Get(j);
        Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting(
            ueNode->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(nrEpcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    nrHelper->AttachToClosestGnb(ueUrlLcNetDev, gnbNetDev);
    nrHelper->AttachToClosestGnb(ueEmbbNetDev, gnbNetDev);
    nrHelper->AttachToClosestGnb(ueMmtcNetDev, gnbNetDev);
    
    Simulator::Schedule(simTime - Seconds(0.01), &PrintSinrStats);
    Simulator::Schedule(simTime - Seconds(0.005), &PrintPowerStats);

    NrEpsBearer urlLcBearer(NrEpsBearer::GBR_CONV_VOICE);
    NrEpsBearer embbBearer(NrEpsBearer::NGBR_VIDEO_TCP_PREMIUM);
    NrEpsBearer mmtcBearer(NrEpsBearer::NGBR_VIDEO_TCP_DEFAULT);

    uint16_t ulPortUrlLc = 5000;
    uint16_t ulPortEmbb = 6000;
    uint16_t ulPortMmtc = 7000;

    ApplicationContainer serverApps;
    ApplicationContainer clientApps;

    for (uint32_t i = 0; i < ueUrlLcContainer.GetN(); ++i) {
        uint16_t port = ulPortUrlLc + i;
        UdpServerHelper server(port);
        serverApps.Add(server.Install(remoteHost));
    }

    for (uint32_t i = 0; i < ueEmbbContainer.GetN(); ++i) {
        uint16_t port = ulPortEmbb + i;
        UdpServerHelper server(port);
        serverApps.Add(server.Install(remoteHost));
    }

    for (uint32_t i = 0; i < ueMmtcContainer.GetN(); ++i) {
        uint16_t port = ulPortMmtc + i;
        UdpServerHelper server(port);
        serverApps.Add(server.Install(remoteHost));
    }
    
    Ptr<UniformRandomVariable> jitterRv = CreateObject<UniformRandomVariable>();
    jitterRv->SetAttribute("Min", DoubleValue(0.0));
    jitterRv->SetAttribute("Max", DoubleValue(1.0));

    for (uint32_t i = 0; i < ueUrlLcContainer.GetN(); ++i) {
        uint16_t port = ulPortUrlLc + i;

        UdpClientHelper client(remoteHostAddr, port);
        client.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
        client.SetAttribute("PacketSize", UintegerValue(udpPacketSizeULL));

        double baseInterval = 1.0 / lambdaULL;
        double jitterFactor = 1.0 + 0.1 * (jitterRv->GetValue() - 0.5); // ±5%
        double interval = baseInterval * jitterFactor;

        client.SetAttribute("Interval", TimeValue(Seconds(interval)));

        ApplicationContainer app = client.Install(ueUrlLcContainer.Get(i));
        app.Start(udpAppStartTime + MilliSeconds(2 * i));
        app.Stop(simTime);

        clientApps.Add(app);
    }

    for (uint32_t i = 0; i < ueEmbbContainer.GetN(); ++i) {
        uint16_t port = ulPortEmbb + i;

        UdpClientHelper client(remoteHostAddr, port);
        client.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
        client.SetAttribute("PacketSize", UintegerValue(udpPacketSizeEMBB));

        double baseInterval = 1.0 / lambdaEMBB;
        double jitterFactor = 1.0 + 0.05 * (jitterRv->GetValue() - 0.5); // ±2.5%
        double interval = baseInterval * jitterFactor;

        client.SetAttribute("Interval", TimeValue(Seconds(interval)));

        ApplicationContainer app = client.Install(ueEmbbContainer.Get(i));

        app.Start(udpAppStartTime + MilliSeconds(5 + 5 * i));
        app.Stop(simTime);

        clientApps.Add(app);
    }

    for (uint32_t i = 0; i < ueMmtcContainer.GetN(); ++i) {
        uint16_t port = ulPortMmtc + i;

        UdpClientHelper client(remoteHostAddr, port);
        client.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
        client.SetAttribute("PacketSize", UintegerValue(udpPacketSizeMMTC));

        double baseInterval = 1.0 / lambdaMMTC;
        double jitterFactor = 1.0 + 0.1 * (jitterRv->GetValue() - 0.5); // ±5%
        double interval = baseInterval * jitterFactor;

        client.SetAttribute("Interval", TimeValue(Seconds(interval)));

        ApplicationContainer app = client.Install(ueMmtcContainer.Get(i));

        app.Start(udpAppStartTime + Seconds(0.1 * i));
        app.Stop(simTime);

        clientApps.Add(app);
    }


    for (uint32_t i = 0; i < ueUrlLcContainer.GetN(); ++i) {
        Ptr<NrEpcTft> tft = Create<NrEpcTft>();
        NrEpcTft::PacketFilter ulpf;
        ulpf.remotePortStart = ulPortUrlLc + i;
        ulpf.remotePortEnd = ulPortUrlLc + i;
        tft->Add(ulpf);
        nrHelper->ActivateDedicatedEpsBearer(ueUrlLcNetDev.Get(i), urlLcBearer, tft);
    }
    
    for (uint32_t i = 0; i < ueEmbbContainer.GetN(); ++i) {
        Ptr<NrEpcTft> tft = Create<NrEpcTft>();
        NrEpcTft::PacketFilter ulpf;
        ulpf.remotePortStart = ulPortEmbb + i;
        ulpf.remotePortEnd = ulPortEmbb + i;
        tft->Add(ulpf);
        nrHelper->ActivateDedicatedEpsBearer(ueEmbbNetDev.Get(i), embbBearer, tft);
    }
    
    for (uint32_t i = 0; i < ueMmtcContainer.GetN(); ++i) {
        Ptr<NrEpcTft> tft = Create<NrEpcTft>();
        NrEpcTft::PacketFilter ulpf;
        ulpf.remotePortStart = ulPortMmtc + i;
        ulpf.remotePortEnd = ulPortMmtc + i;
        tft->Add(ulpf);
        nrHelper->ActivateDedicatedEpsBearer(ueMmtcNetDev.Get(i), mmtcBearer, tft);
    }

    serverApps.Start(udpAppStartTime);
    serverApps.Stop(simTime);
    clientApps.Start(udpAppStartTime);
    clientApps.Stop(simTime);

    FlowMonitorHelper flowmonHelper;
    NodeContainer endpointNodes;
    endpointNodes.Add(remoteHost);
    endpointNodes.Add(ueNodes);
    
    Ptr<FlowMonitor> monitor = flowmonHelper.Install(endpointNodes);
    monitor->SetAttribute("DelayBinWidth", DoubleValue(0.001));
    monitor->SetAttribute("JitterBinWidth", DoubleValue(0.001));
    monitor->SetAttribute("PacketSizeBinWidth", DoubleValue(20));

    NS_LOG_INFO("Starting optimized UPLINK simulation...");
    Simulator::Stop(simTime);
    Simulator::Run();

    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmonHelper.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();

    double totalThroughputUrlLc = 0.0;
    double totalThroughputEmbb = 0.0;
    double totalThroughputMmtc = 0.0;
    double totalDelayUrlLc = 0.0;
    double totalDelayEmbb = 0.0;
    double totalDelayMmtc = 0.0;
    double totalPlrUrlLc = 0.0;
    double totalPlrEmbb = 0.0;
    double totalPlrMmtc = 0.0;
    
    double totalSinrUrlLc = 0.0;
    double totalSinrEmbb = 0.0;
    double totalSinrMmtc = 0.0;
    
    double totalDistanceUrlLc = 0.0;
    double totalDistanceEmbb = 0.0;
    double totalDistanceMmtc = 0.0;
    
    double totalTxPowerUrlLc = 0.0;
    double totalTxPowerEmbb = 0.0;
    double totalTxPowerMmtc = 0.0;
    
    double totalRxPowerUrlLc = 0.0;
    double totalRxPowerEmbb = 0.0;
    double totalRxPowerMmtc = 0.0;
    
    double totalEnergyUrlLc = 0.0;
    double totalEnergyEmbb = 0.0;
    double totalEnergyMmtc = 0.0;
    
    int urlLcFlows = 0, embbFlows = 0, mmtcFlows = 0;

    double flowDuration = (simTime - udpAppStartTime).GetSeconds();
    
    std::ofstream& outFile = outputFiles.back();

    for (const auto& flow : stats) {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(flow.first);
        
        uint32_t sourceFirstOctet = (t.sourceAddress.Get() >> 24) & 0xFF;
        uint32_t destFirstOctet = (t.destinationAddress.Get() >> 24) & 0xFF;
    
        if (sourceFirstOctet != 7 || destFirstOctet != 1) {
            continue;
        }
        
        double throughput = 0.0;
        double delay = 0.0;
        double jitter = 0.0;
        double plr = 0.0;
        double sliceBW = 0.0;
        uint16_t sliceNumerology = 0;
        
        if (flow.second.rxPackets > 0 && flowDuration > 0) {
            throughput = (flow.second.rxBytes * 8.0) / flowDuration / 1e6;
            delay = flow.second.delaySum.GetSeconds() / flow.second.rxPackets;
            if (flow.second.rxPackets > 1)
                jitter = flow.second.jitterSum.GetSeconds() / (flow.second.rxPackets - 1);
        }
        
        if (flow.second.txPackets > 0) {
            plr = ((flow.second.txPackets - flow.second.rxPackets) * 100.0) / flow.second.txPackets;
        }

        std::string sliceType;
        double sinrDb = 0.0;
        double distance = 0.0;
        uint32_t servingGnbId = 0;
        double avgTxPower = 0.0;
        double avgRxPower = 0.0;
        double totalEnergy = 0.0;
        uint32_t globalUeId = 0;
        
        auto ipIt = ipToUeInfoMap.find(t.sourceAddress);
        if (ipIt != ipToUeInfoMap.end()) {
            GlobalUeInfo* ueInfo = ipIt->second;
            globalUeId = ueInfo->globalUeId;
            sinrDb = GetLatestSinr(ueInfo->assignedNodeId);
            avgTxPower = GetAverageTxPower(ueInfo->assignedNodeId);
            avgRxPower = GetAverageRxPower(ueInfo->assignedNodeId);
            totalEnergy = GetTotalEnergyConsumption(ueInfo->assignedNodeId);
            distance = ueInfo->distanceToGnb;
            servingGnbId = ueInfo->servingGnbId;
            
            switch (ueInfo->sliceType) {
                case 0: sliceType = "URLLC"; break;
                case 1: sliceType = "eMBB"; break;
                case 2: sliceType = "mMTC"; break;
            }
        } else {
            for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
                Ptr<Node> ueNode = ueNodes.Get(i);
                Ptr<Ipv4> ipv4 = ueNode->GetObject<Ipv4>();
                Ipv4Address ueAddr = ipv4->GetAddress(1, 0).GetLocal();
                
                if (ueAddr == t.sourceAddress) {
                    uint32_t nodeId = ueNode->GetId();
                    sinrDb = GetLatestSinr(nodeId);
                    avgTxPower = GetAverageTxPower(nodeId);
                    avgRxPower = GetAverageRxPower(nodeId);
                    totalEnergy = GetTotalEnergyConsumption(nodeId);
                    
                    auto nodeIt = nodeIdToUeInfoMap.find(nodeId);
                    if (nodeIt != nodeIdToUeInfoMap.end()) {
                        globalUeId = nodeIt->second->globalUeId;
                        distance = nodeIt->second->distanceToGnb;
                        servingGnbId = nodeIt->second->servingGnbId;
                        switch (nodeIt->second->sliceType) {
                            case 0: sliceType = "URLLC"; break;
                            case 1: sliceType = "eMBB"; break;
                            case 2: sliceType = "mMTC"; break;
                        }
                    }
                    break;
                }
            }
        }
                
        if (sliceType == "URLLC") {
            sliceBW = bandwidthBand1 / 1e6;
            sliceNumerology = numerologyBwp1;
            totalThroughputUrlLc += throughput;
            totalDelayUrlLc += delay;
            totalPlrUrlLc += plr;
            totalSinrUrlLc += sinrDb;
            totalDistanceUrlLc += distance;
            totalTxPowerUrlLc += avgTxPower;
            totalRxPowerUrlLc += avgRxPower;
            totalEnergyUrlLc += totalEnergy;
            urlLcFlows++;
        }
        else if (sliceType == "eMBB") {
            sliceBW = bandwidthBand2 / 1e6;
            sliceNumerology = numerologyBwp2;
            totalThroughputEmbb += throughput;
            totalDelayEmbb += delay;
            totalPlrEmbb += plr;
            totalSinrEmbb += sinrDb;
            totalDistanceEmbb += distance;
            totalTxPowerEmbb += avgTxPower;
            totalRxPowerEmbb += avgRxPower;
            totalEnergyEmbb += totalEnergy;
            embbFlows++;
        }
        else if (sliceType == "mMTC") {
            sliceBW = bandwidthBand3 / 1e6;
            sliceNumerology = numerologyBwp3;
            totalThroughputMmtc += throughput;
            totalDelayMmtc += delay;
            totalPlrMmtc += plr;
            totalSinrMmtc += sinrDb;
            totalDistanceMmtc += distance;
            totalTxPowerMmtc += avgTxPower;
            totalRxPowerMmtc += avgRxPower;
            totalEnergyMmtc += totalEnergy;
            mmtcFlows++;
        }
        else {
            sliceType = "UNKNOWN";
            continue;
        }

        NS_LOG_UNCOND("UL Flow " << flow.first << " (Global UE " << globalUeId 
                      << ", " << t.sourceAddress << ":" << t.sourcePort
                      << " -> " << t.destinationAddress << ":" << t.destinationPort << ")");
        NS_LOG_UNCOND("  Slice: " << sliceType << " (μ=" << sliceNumerology << ", BW: " << sliceBW << " MHz)");
        NS_LOG_UNCOND("  Tx Packets: " << flow.second.txPackets);
        NS_LOG_UNCOND("  Rx Packets: " << flow.second.rxPackets);
        NS_LOG_UNCOND("  Throughput: " << throughput << " Mbps");
        NS_LOG_UNCOND("  Avg Delay: " << delay * 1000 << " ms");
        NS_LOG_UNCOND("  Avg Jitter: " << jitter * 1000 << " ms");
        NS_LOG_UNCOND("  PLR: " << plr << " %");
        NS_LOG_UNCOND("  SINR: " << sinrDb << " dB");
        NS_LOG_UNCOND("  Distance to gNB: " << distance << " m");
        NS_LOG_UNCOND("  Serving gNB ID: " << servingGnbId);
        NS_LOG_UNCOND("  Avg TX Power: " << avgTxPower << " dBm");
        NS_LOG_UNCOND("  Avg RX Power: " << avgRxPower << " dBm");
        NS_LOG_UNCOND("  Total Energy: " << totalEnergy << " J");

        outFile << iteration << "," 
                << globalUeId << "," 
                << flow.first << "," 
                << t.sourceAddress << "," 
                << t.destinationAddress << ","
                << t.sourcePort << "," 
                << t.destinationPort << ","
                << (int)t.protocol << ","
                << flow.second.txPackets << ","
                << flow.second.rxPackets << ","
                << flow.second.txBytes << ","
                << flow.second.rxBytes << ","
                << throughput << ","
                << delay << ","
                << jitter << ","
                << plr << ","
                << sliceType << ","
                << sliceNumerology << ","
                << sliceBW << ","
                << "UL" << ","
                << sinrDb << ","
                << distance << ","
                << servingGnbId << ","
                << avgTxPower << ","
                << avgRxPower << ","
                << totalEnergy << "\n";
    }
    
    NS_LOG_UNCOND("\n=== SLICE PERFORMANCE SUMMARY ===");
    NS_LOG_UNCOND("URLLC Slice (μ=" << numerologyBwp1 << ", " << bandwidthBand1/1e6 << " MHz):");
    NS_LOG_UNCOND("  Flows: " << urlLcFlows);
    NS_LOG_UNCOND("  Total Throughput: " << totalThroughputUrlLc << " Mbps");
    NS_LOG_UNCOND("  Avg Throughput per UE: " << (urlLcFlows > 0 ? totalThroughputUrlLc/urlLcFlows : 0) << " Mbps");
    NS_LOG_UNCOND("  Avg Delay: " << (urlLcFlows > 0 ? (totalDelayUrlLc/urlLcFlows)*1000 : 0) << " ms");
    NS_LOG_UNCOND("  Avg PLR: " << (urlLcFlows > 0 ? totalPlrUrlLc/urlLcFlows : 0) << " %");
    NS_LOG_UNCOND("  Avg SINR: " << (urlLcFlows > 0 ? totalSinrUrlLc/urlLcFlows : 0) << " dB");
    NS_LOG_UNCOND("  Avg Distance to gNB: " << (urlLcFlows > 0 ? totalDistanceUrlLc/urlLcFlows : 0) << " m");
    NS_LOG_UNCOND("  Avg TX Power: " << (urlLcFlows > 0 ? totalTxPowerUrlLc/urlLcFlows : 0) << " dBm");
    NS_LOG_UNCOND("  Avg RX Power: " << (urlLcFlows > 0 ? totalRxPowerUrlLc/urlLcFlows : 0) << " dBm");
    NS_LOG_UNCOND("  Total Energy: " << totalEnergyUrlLc << " J");
    NS_LOG_UNCOND("  Avg Energy per UE: " << (urlLcFlows > 0 ? totalEnergyUrlLc/urlLcFlows : 0) << " J");
    
    NS_LOG_UNCOND("\neMBB Slice (μ=" << numerologyBwp2 << ", " << bandwidthBand2/1e6 << " MHz):");
    NS_LOG_UNCOND("  Flows: " << embbFlows);
    NS_LOG_UNCOND("  Total Throughput: " << totalThroughputEmbb << " Mbps");
    NS_LOG_UNCOND("  Avg Throughput per UE: " << (embbFlows > 0 ? totalThroughputEmbb/embbFlows : 0) << " Mbps");
    NS_LOG_UNCOND("  Avg Delay: " << (embbFlows > 0 ? (totalDelayEmbb/embbFlows)*1000 : 0) << " ms");
    NS_LOG_UNCOND("  Avg PLR: " << (embbFlows > 0 ? totalPlrEmbb/embbFlows : 0) << " %");
    NS_LOG_UNCOND("  Avg SINR: " << (embbFlows > 0 ? totalSinrEmbb/embbFlows : 0) << " dB");
    NS_LOG_UNCOND("  Avg Distance to gNB: " << (embbFlows > 0 ? totalDistanceEmbb/embbFlows : 0) << " m");
    NS_LOG_UNCOND("  Avg TX Power: " << (embbFlows > 0 ? totalTxPowerEmbb/embbFlows : 0) << " dBm");
    NS_LOG_UNCOND("  Avg RX Power: " << (embbFlows > 0 ? totalRxPowerEmbb/embbFlows : 0) << " dBm");
    NS_LOG_UNCOND("  Total Energy: " << totalEnergyEmbb << " J");
    NS_LOG_UNCOND("  Avg Energy per UE: " << (embbFlows > 0 ? totalEnergyEmbb/embbFlows : 0) << " J");
    
    NS_LOG_UNCOND("\nmMTC Slice (μ=" << numerologyBwp3 << ", " << bandwidthBand3/1e6 << " MHz):");
    NS_LOG_UNCOND("  Flows: " << mmtcFlows);
    NS_LOG_UNCOND("  Total Throughput: " << totalThroughputMmtc << " Mbps");
    NS_LOG_UNCOND("  Avg Throughput per UE: " << (mmtcFlows > 0 ? totalThroughputMmtc/mmtcFlows : 0) << " Mbps");
    NS_LOG_UNCOND("  Avg Delay: " << (mmtcFlows > 0 ? (totalDelayMmtc/mmtcFlows)*1000 : 0) << " ms");
    NS_LOG_UNCOND("  Avg PLR: " << (mmtcFlows > 0 ? totalPlrMmtc/mmtcFlows : 0) << " %");
    NS_LOG_UNCOND("  Avg SINR: " << (mmtcFlows > 0 ? totalSinrMmtc/mmtcFlows : 0) << " dB");
    NS_LOG_UNCOND("  Avg Distance to gNB: " << (mmtcFlows > 0 ? totalDistanceMmtc/mmtcFlows : 0) << " m");
    NS_LOG_UNCOND("  Avg TX Power: " << (mmtcFlows > 0 ? totalTxPowerMmtc/mmtcFlows : 0) << " dBm");
    NS_LOG_UNCOND("  Avg RX Power: " << (mmtcFlows > 0 ? totalRxPowerMmtc/mmtcFlows : 0) << " dBm");
    NS_LOG_UNCOND("  Total Energy: " << totalEnergyMmtc << " J");
    NS_LOG_UNCOND("  Avg Energy per UE: " << (mmtcFlows > 0 ? totalEnergyMmtc/mmtcFlows : 0) << " J");

    Simulator::Destroy();
    NS_LOG_INFO("Batch " << batchNumber << ", Iteration " << iteration << " completed.");
}

int
main(int argc, char* argv[]) {
    seeds = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    totalIterations = seeds.size();
    
    CreateResultsDirectory();
    
    InitializeGlobalUeInfo();
    
    uint32_t totalBatches = totalUes / batchSize;
    if (totalUes % batchSize != 0) {
        totalBatches++;
    }
    NS_LOG_UNCOND("\nGENERAL CONFIGURATION");
    NS_LOG_UNCOND("  Total UEs: " << totalUes);
    NS_LOG_UNCOND("  Batch size: " << batchSize);
    NS_LOG_UNCOND("  Total batches: " << totalBatches);
    NS_LOG_UNCOND("  Iterations per batch: " << totalIterations);
    
    for (currentBatch = 0; currentBatch < totalBatches; currentBatch++) {
        NS_LOG_UNCOND("\n******************************************************");
        NS_LOG_UNCOND("Progress: Processing batch " << currentBatch + 1 << "/" << totalBatches);
        NS_LOG_UNCOND("******************************************************");
        
        std::vector<GlobalUeInfo*> batchUes = GetUesForBatch(currentBatch);
        
        if (CheckBatchFileComplete(currentBatch)) {
            NS_LOG_UNCOND("✓ Batch " << currentBatch << " already completed. Skipping to next batch.");
            continue;
        }
        
        uint32_t startIteration = FindStartingIteration(currentBatch);
        
        if (startIteration >= totalIterations) {
            NS_LOG_UNCOND("✓ Batch " << currentBatch << " already has all " << totalIterations << " iterations. Skipping.");
            continue;
        }
        
        OpenBatchOutputFile();
        
        NS_LOG_UNCOND("Resuming from iteration " << startIteration << " - batch " << currentBatch);
        
        for (currentIteration = startIteration; currentIteration < totalIterations; currentIteration++) {
            RunBatchSimulation(currentBatch, currentIteration, batchUes);
        }
        
        if (outputFiles.back().is_open()) {
            outputFiles.back().close();
            outputFiles.pop_back();
        }
        
        NS_LOG_UNCOND("\n✓ Completed Batch " << currentBatch + 1 << " with " << totalIterations << " iterations");
    }

    return 0;
}
