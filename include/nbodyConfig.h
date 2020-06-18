#include <iostream>
#include <fstream>

struct ConfigData {
	int particleCount;
	int totalIterations;
	int save_Image_Every_Xth_Iteration;
	float timestep;
	float minRandBodyMass;
	float maxRandBodyMass;
	float minRadius;
	float maxRadius;
	float growthRate;
	int imgWidth;
	int imgHeight;
	int fieldWidth;
	int fieldHeight;
	std::string imagePath;
};


ConfigData parseConfigFile(const std::string& filepath) {
	ConfigData conf;
	std::ifstream configFileStream(filepath);
	if (!configFileStream.is_open()) {
		std::cout << "Error opening config file! Exiting..." << std::endl;
		exit(1);
	}

	std::string line;
	std::string variableName; //name of the variable the line in the config will modify
	size_t delimPos;
	while (std::getline(configFileStream, line)) {
		delimPos = line.find("=");
		variableName = line.substr(0, delimPos);
		if (variableName.compare("particleCount") == 0)
		{
			int particleCount;
			try {
				particleCount = std::stoi(line.substr(delimPos + 1));
			} catch (std::exception const &e) {
				std::cout << "particleCount invalid value: " << e.what()
						<< std::endl;
				exit(1);
			}
			std::cout << "particleCount=" << particleCount
					<< std::endl;
			conf.particleCount = particleCount;
		}
		else if (variableName.compare("totalIterations") == 0)
		{
			int iterations;
			try {
				iterations = std::stoi(line.substr(delimPos + 1));
			} catch (std::exception const &e) {
				std::cout << "totalIterations invalid value: " << e.what()
						<< std::endl;
				exit(1);
			}
			std::cout << "totalIterations=" << iterations
					<< std::endl;
			conf.totalIterations = iterations;
		}
		else if (variableName.compare("save_Image_Every_Xth_Iteration") == 0)
		{
			int saveAt;
			try {
				saveAt = std::stoi(line.substr(delimPos + 1));
			} catch (std::exception const &e) {
				std::cout << "save_Image_Every_Xth_Iteration invalid value: "
						<< e.what() << std::endl;
				exit(1);
			}
			std::cout << "save_Image_Every_Xth_Iteration="
					<< saveAt << std::endl;
			conf.save_Image_Every_Xth_Iteration = saveAt;
		}
		else if (variableName.compare("timestep") == 0)
		{
			float timestep;
			try {
				timestep = std::stof(line.substr(delimPos + 1));
			} catch (std::exception const &e) {
				std::cout << "timestep invalid value: " << e.what()
						<< std::endl;
				exit(1);
			}
			std::cout << "timestep=" << timestep << std::endl;
			conf.timestep = timestep;
		}
		else if (variableName.compare("minRandBodyMass") == 0)
		{
			float minRandBodyMass;
			try {
				minRandBodyMass = std::stof(line.substr(delimPos + 1));
			} catch (std::exception const &e) {
				std::cout << "minRandBodyMass invalid value: " << e.what()
						<< std::endl;
				exit(1);
			}
			std::cout << "minRandBodymass=" << minRandBodyMass
					<< std::endl;
			conf.minRandBodyMass = minRandBodyMass;
		}
		else if (variableName.compare("maxRandBodyMass") == 0)
		{
			float maxRandBodyMass;
			try {
				maxRandBodyMass = std::stof(line.substr(delimPos + 1));
			} catch (std::exception const &e) {
				std::cout << "maxRandBodyMass invalid value: " << e.what()
						<< std::endl;
				exit(1);
			}
			std::cout << "maxRandBodyMass=" << maxRandBodyMass
					<< std::endl;
			conf.maxRandBodyMass = maxRandBodyMass;
		}
		else if (variableName.compare("minRadius") == 0)
		{
			float minRadius;
			try {
				minRadius = std::stof(line.substr(delimPos + 1));
			} catch (std::exception const &e) {
				std::cout << "minRadius invalid value: " << e.what()
						<< std::endl;
				exit(1);
			}
			std::cout << "minRadius=" << minRadius
					<< std::endl;
			conf.minRadius = minRadius;
		}
		else if (variableName.compare("maxRadius") == 0)
		{
			float maxRadius;
			try {
				maxRadius = std::stof(line.substr(delimPos + 1));
			} catch (std::exception const &e) {
				std::cout << "maxRadius invalid value: " << e.what()
						<< std::endl;
				exit(1);
			}
			std::cout << "maxRadius=" << maxRadius
					<< std::endl;
			conf.maxRadius = maxRadius;
		}
		else if (variableName.compare("imgWidth") == 0)
		{
			int imgWidth;
			try {
				imgWidth = std::stoi(line.substr(delimPos + 1));
			} catch (std::exception const &e) {
				std::cout << "imgWidth invalid value: " << e.what()
						<< std::endl;
				exit(1);
			}
			std::cout << "imgWidth=" << imgWidth << std::endl;
			conf.imgWidth = imgWidth;
		}
		else if (variableName.compare("imgHeight") == 0)
		{
			int imgHeight;
			try {
				imgHeight = std::stoi(line.substr(delimPos + 1));
			} catch (std::exception const &e) {
				std::cout << "imgHeight invalid value: " << e.what()
						<< std::endl;
				exit(1);
			}
			std::cout << "imgHeight=" << imgHeight << std::endl;
			conf.imgHeight = imgHeight;
		}
		else if (variableName.compare("fieldWidth") == 0)
		{
			int fieldWidth;
			try {
				fieldWidth = std::stoi(line.substr(delimPos + 1));
			} catch (std::exception const &e) {
				std::cout << "fieldWidth invalid value: " << e.what()
						<< std::endl;
				exit(1);
			}
			std::cout << "fieldWidth=" << fieldWidth
					<< std::endl;
			conf.fieldWidth = fieldWidth;
		}
		else if (variableName.compare("fieldHeight") == 0)
		{
			int fieldHeight;
			try {
				fieldHeight = std::stoi(line.substr(delimPos + 1));
			} catch (std::exception const &e) {
				std::cout << "fieldHeight invalid value: " << e.what()
						<< std::endl;
				exit(1);
			}
			std::cout << "fieldHeight=" << fieldHeight
					<< std::endl;
			conf.fieldHeight = fieldHeight;
		}
		else if (variableName.compare("imagePath") == 0)
		{
			std::string imagePath = line.substr(delimPos + 1);
			std::cout << "imagePath=" << imagePath
					<< std::endl;
			conf.imagePath = imagePath;
		}
		else if (variableName.compare("radiusGrowthRate") == 0)
		{
			float growthRate;
			try {
				growthRate = std::stof(line.substr(delimPos + 1));
			} catch (std::exception const &e) {
				std::cout << "growthRate invalid value: " << e.what()
						<< std::endl;
				exit(1);
			}
			std::cout << "growthRate=" << growthRate
					<< std::endl;
			conf.growthRate = growthRate;
		}
		else {
			std::cout << "Invalid variable: " << variableName << std::endl;
		}
	}
	return conf;
}
