"""
Для геокодирования используется открытый Nominatim API, который использует данные из OpenStreetMap (OSM)

https://nominatim.org/release-docs/develop/api/Overview/
"""

import requests


class GeocoderOSM:
    def __init__(self):
        self.url = 'https://nominatim.openstreetmap.org/search?'

    def get_coordinates(self, query, format='json', limit=1):
        """
        Метод для прямого геокодирования

        :param query: Текстовый запрос, например, Санкт-Петербург
        :param format: xml | json | jsonv2 | geojson | geocodejson
        :param limit: по умолчанию ожидаем 1 результат
        :return: dict: address: полное наименование адреса; address_type: тип адреса; lat: широта; lon: долгота
        """
        url = f'{self.url}q={query}&format={format}&limit={limit}'
        req = requests.get(url=url, verify=False).json()

        if req[0]:
            address = req[0]['display_name']
            address_type = req[0]['type']
            lat = req[0]['lat']
            lon = req[0]['lon']
            return {'address': address, 'address_type': address_type, 'lat': lat, 'lon': lon}
        else:
            return {'address': 'NO ADDRESS', 'address_type': 'NO TYPE', 'lat': -1, 'lon': -1}
